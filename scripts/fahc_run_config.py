"""Validate a run and record provenance without importing a model or CUDA."""

import argparse
import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.fahc_config import ALGORITHM_VERSION, experiment_signature, parse_bool, validate_args


def parser_without_models():
    path = ROOT / "src/main.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef)
                    and node.name == "build_arg_parser")
    scope = {"argparse": argparse, "parse_bool": parse_bool}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), scope)
    return scope["build_arg_parser"]()


def fingerprint(paths):
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(b"\0")
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


def main():
    args = parser_without_models().parse_args()
    validate_args(args)
    dataset = ROOT / "data" / args.dataset
    data_files = [dataset / name for name in
                  ("train.txt", "valid.txt", "test.txt", "stat.txt")]
    if args.add_static_graph:
        data_files.append(dataset / "e-w-graph.txt")
    for path in data_files:
        if not path.is_file():
            raise FileNotFoundError(path)
    cache = list((dataset / "history_snap_v2").glob("snap_*.pkl"))
    if not cache:
        raise ValueError("prepare a consistent history_snap_v2 cache before running experiments")
    snapshot_count = 0
    for name in ("train.txt", "valid.txt", "test.txt"):
        previous = None
        with (dataset / name).open() as stream:
            for line in stream:
                values = line.split()
                if not values:
                    continue
                time = int(values[3])
                if time != previous:
                    snapshot_count += 1
                    previous = time
    available = {path.name for path in cache}
    missing = {f"snap_{index}.pkl" for index in range(snapshot_count)} - available
    if missing:
        raise ValueError(f"incomplete historical cache: {len(missing)} snapshots missing")
    source = [path for directory in ("src", "diffusion", "rgcn")
              for path in (ROOT / directory).rglob("*.py")]
    source.extend(Path(__file__).parent.glob("*.py"))
    source.extend(Path(__file__).parent.glob("*.sh"))
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    versions = {}
    for package in ("torch", "dgl", "numpy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            raise RuntimeError(f"missing dependency: {package}")
    print(json.dumps({
        "settings": vars(args), "settings_hash": experiment_signature(args),
        "algorithm_version": ALGORITHM_VERSION,
        "git_revision": revision, "source_sha256": fingerprint(source),
        "data_sha256": fingerprint(data_files), "history_sha256": fingerprint(cache),
        "history_file_count": len(cache), "snapshot_count": snapshot_count,
        "python": sys.version,
        "dependencies": versions,
    }, indent=2))


if __name__ == "__main__":
    main()
