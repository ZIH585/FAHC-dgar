#!/usr/bin/env bash
set -euo pipefail
DGAR_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DGAR_INTERPRETER="${DGAR_PYTHON:-python}"
cd "$DGAR_ROOT"
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
"$DGAR_INTERPRETER" -B -W ignore::FutureWarning -W ignore::ResourceWarning \
  -m unittest discover -s tests -v
"$DGAR_INTERPRETER" -B src/main.py --help > /dev/null
"$DGAR_INTERPRETER" -B scripts/evaluate_rter.py --help > /dev/null
bash -n scripts/run_rter_experiment.sh scripts/run_rter_evaluation.sh
"$DGAR_INTERPRETER" -B - <<'PY'
from pathlib import Path
import py_compile
import tempfile
sources = [p for folder in ('src', 'diffusion', 'rgcn', 'tests', 'scripts')
           for p in Path(folder).glob('*.py')]
with tempfile.TemporaryDirectory() as directory:
    for index, path in enumerate(sources):
        py_compile.compile(str(path), cfile=str(Path(directory) / f'{index}.pyc'), doraise=True)
print(f'Compiled {len(sources)} Python files; --help and CPU checks passed.')
PY
