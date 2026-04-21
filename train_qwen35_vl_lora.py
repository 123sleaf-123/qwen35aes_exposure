import argparse
import inspect
import json
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import AutoProcessor, BitsAndBytesConfig, Trainer, TrainingArguments

try:
    from transformers import AutoModelForImageTextToText
except ImportError:
    from transformers import AutoModelForVision2Seq as AutoModelForImageTextToText


def ensure_set_submodule_compatibility():
    if hasattr(nn.Module, "set_submodule"):
        return

    def set_submodule(self, target: str, module: nn.Module):
        if not target:
            raise ValueError("target must be a non-empty module path")
        parts = target.split(".")
        parent = self
        for part in parts[:-1]:
            parent = parent.get_submodule(part)
        setattr(parent, parts[-1], module)

    nn.Module.set_submodule = set_submodule


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA SFT for Qwen3.5-VL on IQA data.")
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--eval-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--num-train-epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="Optional base directory used to resolve relative image paths stored in JSONL records.",
    )
    return parser.parse_args()


class JsonlMultimodalDataset(Dataset):
    def __init__(self, file_path: str, dataset_root: str | None = None):
        self.file_path = Path(file_path)
        self.dataset_root = Path(dataset_root).resolve() if dataset_root else self.file_path.resolve().parent
        self.records = []
        with self.file_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                image_path = Path(record["image_path"])
                if not image_path.is_absolute():
                    record["image_path"] = str((self.dataset_root / image_path).resolve())

                user_content = record.get("messages", [{}])[0].get("content", [])
                for item in user_content:
                    if item.get("type") != "image":
                        continue
                    item_path = Path(item["image"])
                    if not item_path.is_absolute():
                        item["image"] = str((self.dataset_root / item_path).resolve())

                self.records.append(record)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


def load_rgb_image(image_path: str) -> Image.Image:
    with Image.open(image_path) as image:
        return image.convert("RGB")


def gather_special_token_ids(processor):
    token_ids = set()
    tokenizer = processor.tokenizer
    candidate_attrs = [
        "image_token",
        "video_token",
        "vision_start_token",
        "vision_end_token",
        "image_start_token",
        "image_end_token",
    ]
    for attr_name in candidate_attrs:
        token = getattr(processor, attr_name, None)
        if isinstance(token, str):
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id is not None and token_id >= 0:
                token_ids.add(token_id)
    return token_ids


class Qwen35VLDataCollator:
    def __init__(self, processor, max_length: int):
        self.processor = processor
        self.max_length = max_length
        self.special_token_ids = gather_special_token_ids(processor)

    def __call__(self, features):
        images = []
        full_texts = []
        prompt_texts = []

        for feature in features:
            image = load_rgb_image(feature["image_path"])
            messages = feature["messages"]
            user_messages = [messages[0]]
            images.append(image)
            full_texts.append(
                self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )
            prompt_texts.append(
                self.processor.apply_chat_template(
                    user_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )

        batch = self.processor(
            text=full_texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        prompt_batch = self.processor(
            text=prompt_texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        labels = batch["input_ids"].clone()
        prompt_lengths = prompt_batch["attention_mask"].sum(dim=1).tolist()
        for row_index, prompt_length in enumerate(prompt_lengths):
            labels[row_index, : int(prompt_length)] = -100

        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100

        for token_id in self.special_token_ids:
            labels[labels == token_id] = -100

        batch["labels"] = labels
        return batch


def build_model_and_processor(args):
    ensure_set_submodule_compatibility()

    quantization_config = None
    model_kwargs = {"trust_remote_code": True}
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = "auto"
    elif torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.bfloat16 if args.bf16 else torch.float16

    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(args.model_name, **model_kwargs)

    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=args.target_modules,
    )
    model = get_peft_model(model, lora_config)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    model.print_trainable_parameters()
    return model, processor


def main():
    args = parse_args()

    train_dataset = JsonlMultimodalDataset(args.train_file, dataset_root=args.dataset_root)
    eval_dataset = JsonlMultimodalDataset(args.eval_file, dataset_root=args.dataset_root)
    model, processor = build_model_and_processor(args)
    collator = Qwen35VLDataCollator(processor=processor, max_length=args.max_length)

    training_kwargs = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.train_batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "save_strategy": "steps",
        "save_total_limit": args.save_total_limit,
        "bf16": args.bf16,
        "fp16": args.fp16,
        "remove_unused_columns": False,
        "report_to": args.report_to,
        "dataloader_num_workers": 2,
        "optim": "paged_adamw_8bit" if args.load_in_4bit else "adamw_torch",
        "lr_scheduler_type": "cosine",
        "gradient_checkpointing": args.gradient_checkpointing,
    }
    training_signature = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in training_signature.parameters:
        training_kwargs["eval_strategy"] = "steps"
    else:
        training_kwargs["evaluation_strategy"] = "steps"

    training_args = TrainingArguments(**training_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()