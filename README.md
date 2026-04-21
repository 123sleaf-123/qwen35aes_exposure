# qwen35vl_train_repo

Lightweight training bundle copied from `score_260310`. Included assets:

- `train_qwen35_vl_lora.py`
- `run_qwen35_vl_lora.sh`
- `qwen35vl_env.yml`
- `create_qwen35vl_env.sh`
- `score_result_qwen35_vl_sft/` with repo-relative image paths

Expected layout:

```
qwen35vl_train_repo/
├── images/
│   └── REED_3_RESIZE_512/
│       ├── training/
│       ├── validation/
│       └── testing/
├── score_result_qwen35_vl_sft/
└── ...training scripts and env files...
```

Place the image tree under `images/REED_3_RESIZE_512` before training. The dataset JSONL files and `manifest.json` now reference this repo-relative image location.
