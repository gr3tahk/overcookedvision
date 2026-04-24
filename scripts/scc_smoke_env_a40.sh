#!/bin/bash -l
#$ -P cs585
#$ -N overcooked_smoke_a40
#$ -l h_rt=00:20:00
#$ -pe omp 4
#$ -l gpus=1
#$ -l gpu_type=A40
#$ -j y
#$ -o logs/$JOB_NAME.$JOB_ID.out

set -euo pipefail

module load miniconda
module load academic-ml/spring-2026
echo "CONDA_BASE=$(conda info --base)"
conda info --envs
conda activate spring-2026-pyt

cd "$SGE_O_WORKDIR"

echo "HOST=$(hostname)"
echo "PWD=$(pwd)"
echo "PYTHON=$(which python)"
python --version

echo "== torch check =="
python - <<'PY'
import sys
print("python:", sys.executable)
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
print("gpu count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

echo "== install project =="
python -m pip install -e .

echo "== tests =="
python -m unittest discover -s tests

echo "== scripted benchmark smoke =="
python benchmark.py \
  --pair scripted-scripted \
  --trials 1 \
  --max-ticks 4 \
  --experiment-output results/smoke_scripted.json

echo "== summary =="
python -m overcooked_benchmark.summarize results/smoke_scripted.json

echo "DONE"
