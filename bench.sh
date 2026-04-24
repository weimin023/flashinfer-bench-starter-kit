git lfs install
git clone https://huggingface.co/datasets/flashinfer-ai/mlsys26-contest
export FIB_DATASET_PATH=/path/to/flashinfer-trace
python scripts/pack_solution.py
modal setup
modal volume create flashinfer-trace
modal volume put flashinfer-trace ./mlsys26-contest
modal run scripts/run_modal.py
