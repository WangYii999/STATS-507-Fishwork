# Persona Dialogue Generation — Final Package

Lightweight, reproducible code to train and evaluate a persona‑aware dialogue generator using FLAN‑T5 small with LoRA and retrieval augmentation.

## Folder Structure
```
final_package/
├─ pipeline/run_stage2.py        # full pipeline (self‑contained)
├─ run.py                        # unified CLI (train/eval)
├─ requirements.txt              # minimal dependencies
├─ README.md                     # how to use
├─ run_all.bat                   # Windows: one‑click eval + images
├─ Makefile                      # make eval2/eval4/images/all
└─ scripts/make_report_images.py # generate report figures
```

The pipeline caches datasets/models under `hf_cache/` (auto‑created). Results go to `outputs/`. Report figures go to `report_images/`.

## Quick Start

1. Python 3.9+（GPU 推荐，CPU 亦可）。安装依赖：
   ```
   pip install -r final_package/requirements.txt
   ```
   若本机未安装 PyTorch，请参考 https://pytorch.org 获取适配的安装命令（CUDA/CPU）。
2. Run evaluation（均衡档）
   ```
   python final_package/run.py --mode eval --run_id beams2 \
     --rag_topk 4 --max_new_tokens 160 --beams 2 --length_penalty 1.0 \
     --outdir outputs/t5-small-lora-longmax --device cuda
   ```
3. Run evaluation（覆盖档）
   ```
   python final_package/run.py --mode eval --run_id beams4pp \
     --rag_topk 4 --max_new_tokens 192 --beams 4 --length_penalty 1.2 \
     --outdir outputs/t5-small-lora-longmax --device cuda
   ```
4. （可选）训练长稳态设定
   ```
   python final_package/run.py --mode train --run_id longmaxpp \
     --n_train 1600 --n_val 320 --n_test 320 --max_steps 2000 --warmup_steps 300 \
     --grad_acc 2 --unfreeze_last 4 --rag_topk 4 --outdir outputs/t5-small-lora-longmaxpp --device cuda
   ```

### 一键运行
- Windows：双击或在终端运行 `final_package\run_all.bat`
- 跨平台：`make all` 或分别执行 `make eval2`、`make eval4`、`make images`

## Outputs
- 摘要：`outputs/stage3_summary_<RUN_ID>.json`
- 案例：`outputs/stage3_cases_<RUN_ID>.jsonl`
- 完成标记：`outputs/stage3_done_<RUN_ID>.txt`

## Report Images
生成报告插图：
```
python final_package/scripts/make_report_images.py
```
输出到 `report_images/`：beam对比、基线对比、训练曲线、结构图等。

## Tips & Troubleshooting
- 无 GPU 时使用 `--device cpu`（速度会变慢）。
- 首次运行会自动下载数据与模型到 `hf_cache/`；如需自定义缓存位置，设置环境变量 `HF_HOME`。
- 若缺包（如 `torch`, `sentencepiece`, `evaluate`），请执行：`pip install -r final_package/requirements.txt`。
- 请勿提交大模型权重与大日志，只提交代码、配置与必要小图。
