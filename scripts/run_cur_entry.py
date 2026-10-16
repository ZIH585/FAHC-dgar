"""Launch the real main.py, optionally offloading autograd saved tensors to CPU."""
import argparse
from contextlib import nullcontext
import runpy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--cpu-offload', action='store_true')
    options, arguments = parser.parse_known_args()
    import torch
    context = torch.autograd.graph.save_on_cpu(pin_memory=False) if options.cpu_offload else nullcontext()
    print(f'Execution CPU saved-tensor offload: {options.cpu_offload}', flush=True)
    sys.argv = [str(ROOT / 'src/main.py'), *arguments]
    with context:
        runpy.run_path(sys.argv[0], run_name='__main__')


if __name__ == '__main__':
    main()
