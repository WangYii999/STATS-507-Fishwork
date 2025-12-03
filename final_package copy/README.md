# Persona Dialogue Generation — Final Package (Reproduction Copy)

This copy contains the full pipeline plus complete experiment artifacts (outputs, checkpoints, figures) for end‑to‑end reproduction.

## Folder Structure
```
final_package copy/
├─ pipeline/run_stage2.py        # full pipeline (self‑contained)
├─ run.py                        # unified CLI (train/eval)
├─ requirements.txt              # minimal dependencies
├─ README.md                     # usage and tips
├─ run_all.bat                   # Windows: one‑click eval + images
├─ Makefile                      # make eval2/eval4/images/all
├─ outputs/                      # summaries, cases, checkpoints, metrics
└─ scripts/make_report_images.py # generate report figures
```

The pipeline caches datasets/models under `hf_cache/` (auto‑created). Results are written to `outputs/`. Report figures are written to `report_images/`.

## Setup
- Python 3.9+ (GPU recommended; CPU supported)
- Install dependencies:
  ```
  pip install -r "final_package copy/requirements.txt"
  ```
- If PyTorch is not installed, follow https://pytorch.org for the proper CUDA/CPU install command.

## Quick Start
1) Balanced decoding (beam=2):
```
python "final_package copy/run.py" --mode eval --run_id beams2 \
  --rag_topk 4 --max_new_tokens 160 --beams 2 --length_penalty 1.0 \
  --outdir "final_package copy/outputs/t5-small-lora-longmax" --device cuda
```
2) Coverage‑oriented decoding (beam=4):
```
python "final_package copy/run.py" --mode eval --run_id beams4pp \
  --rag_topk 4 --max_new_tokens 192 --beams 4 --length_penalty 1.2 \
  --outdir "final_package copy/outputs/t5-small-lora-longmax" --device cuda
```
3) Optional training (long‑steady setup):
```
python "final_package copy/run.py" --mode train --run_id longmaxpp \
  --n_train 1600 --n_val 320 --n_test 320 --max_steps 2000 --warmup_steps 300 \
  --grad_acc 2 --unfreeze_last 4 --rag_topk 4 --outdir "final_package copy/outputs/t5-small-lora-longmaxpp" --device cuda
```

### One‑Click
- Windows: run `final_package copy\run_all.bat`
- Cross‑platform: `make all` or run `make eval2`, `make eval4`, `make images`

## Outputs
- Summary: `final_package copy/outputs/stage3_summary_<RUN_ID>.json`
- Cases: `final_package copy/outputs/stage3_cases_<RUN_ID>.jsonl`
- Done flag: `final_package copy/outputs/stage3_done_<RUN_ID>.txt`
- Checkpoints/weights: `final_package copy/outputs/t5-small-lora-longmax/`

## Report Images
Generate figures:
```
python "final_package copy/scripts/make_report_images.py"
```
Figures are written to `final_package copy/report_images/`: beam comparison, baseline comparison, training curves, beam 1/2/4, length 48 vs 96.

## Tips & Troubleshooting
- Use `--device cpu` when GPU is unavailable (slower).
- First run downloads datasets/models into `hf_cache/`; set `HF_HOME` to customize cache location.
- If packages are missing (e.g., `torch`, `sentencepiece`, `evaluate`), run `pip install -r "final_package copy/requirements.txt"`.
- For version control, use Git LFS to track `outputs/`, checkpoints and weights.

## Reproducibility
Code, configs and figure scripts are designed for end‑to‑end reproduction. Repository link:
```
https://github.com/WangYii999/STATS-507-Fishwork
```
