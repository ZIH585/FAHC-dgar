"""Resume an existing PGR control run with autograd activation CPU offload.

The model sources and checkpoint provenance stay unchanged. This wrapper only
changes where autograd stores saved tensors, using full precision throughout.
"""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def build_resume_args(config, run_dir, gpu, data_root):
    from src.cli import build_arg_parser, validate_args
    from src.runtime import settings

    args = build_arg_parser().parse_args([])
    saved = config["settings"]
    unknown = set(saved) - set(vars(args))
    if unknown:
        raise ValueError(f"Unrecognized saved settings: {sorted(unknown)}")
    vars(args).update(saved)
    if args.pgr_mode not in ("original", "mean", "shared", "mean-shared"):
        raise ValueError("This resume wrapper supports Original/mean/Shared controls only")
    args.gpu = gpu
    args.data_root = Path(data_root).resolve()
    args.run_dir = Path(run_dir).resolve()
    args.resume = args.run_dir / "latest.pt"
    args.test = False
    args.final_validation = True
    args.pgr_diagnostics = True
    validate_args(args)
    if settings(args) != saved:
        raise ValueError("Saved scientific settings cannot be reproduced by this parser")
    return args


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--data-root", type=Path,
                        default=Path(os.environ.get("DGAR_DATA_ROOT", ROOT / "data")))
    parser.add_argument("--dry-run", action="store_true",
                        help="Verify saved config/source/data/cache without training or CUDA")
    options = parser.parse_args(argv)
    run_dir = options.run_dir.resolve()
    config = json.loads((run_dir / "config.json").read_text())
    if not (run_dir / "latest.pt").is_file():
        raise FileNotFoundError(run_dir / "latest.pt")
    args = build_resume_args(config, run_dir, options.gpu, options.data_root)

    # This check hashes files only; it does not load data or construct a model.
    from src.runtime import prepare_run
    prepare_run(args)
    if args._provenance != config:
        raise ValueError("Config/source/data/cache differ from the interrupted run")
    print(f"Resume: {args.resume}", flush=True)
    print(f"Mode: {args.pgr_mode}; dataset: {args.dataset}; seed: {args.seed}; GPU: {args.gpu}", flush=True)
    print("Execution: full-precision autograd save_on_cpu(pin_memory=False)", flush=True)
    if options.dry_run:
        print("Validated. No training or CUDA initialization performed.", flush=True)
        return

    import torch
    from src.main import run_experiment
    # Keep this context around all predictor, diffusion and guidance backwards.
    # No recomputation, minibatching, AMP, RNG reseeding or loss rescaling.
    with torch.autograd.graph.save_on_cpu(pin_memory=False):
        run_experiment(args)


if __name__ == "__main__":
    main()
