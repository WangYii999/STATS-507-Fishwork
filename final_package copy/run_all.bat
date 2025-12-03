@echo off
setlocal
set HF_HOME=%CD%\final_package copy\hf_cache
echo HF_HOME=%HF_HOME%

.venv\Scripts\python.exe "final_package copy\run.py" --mode eval --run_id beams2 --rag_topk 4 --max_new_tokens 160 --beams 2 --length_penalty 1.0 --outdir "final_package copy\outputs\t5-small-lora-longmax" --device cuda
.venv\Scripts\python.exe "final_package copy\run.py" --mode eval --run_id beams4pp --rag_topk 4 --max_new_tokens 192 --beams 4 --length_penalty 1.2 --outdir "final_package copy\outputs\t5-small-lora-longmax" --device cuda
.venv\Scripts\python.exe "final_package copy\scripts\make_report_images.py"

echo Done. See final_package copy\outputs\ and final_package copy\report_images\.
endlocal
