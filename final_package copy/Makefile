PY=python
HF_HOME=$(PWD)/final_package/hf_cache

.PHONY: all eval2 eval4 images

all: eval2 eval4 images

eval2:
	HF_HOME=$(HF_HOME) $(PY) final_package/run.py --mode eval --run_id beams2 --rag_topk 4 --max_new_tokens 160 --beams 2 --length_penalty 1.0 --outdir final_package/outputs/t5-small-lora-longmax --device cuda

eval4:
	HF_HOME=$(HF_HOME) $(PY) final_package/run.py --mode eval --run_id beams4pp --rag_topk 4 --max_new_tokens 192 --beams 4 --length_penalty 1.2 --outdir final_package/outputs/t5-small-lora-longmax --device cuda

images:
	$(PY) final_package/scripts/make_report_images.py
