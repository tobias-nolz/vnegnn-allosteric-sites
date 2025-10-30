#!/usr/bin/env python3
"""
Portable Python wrapper for scripts/process_data.sh
Runs generate_esm_embeddings.py and extract_binding_info.py for every folder under data/data
"""
import argparse
from pathlib import Path
import subprocess
import sys
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "vnegnn" / "scripts"


def run_command(cmd, cwd=None):
    print("Running:", " ".join(cmd))
    res = subprocess.run(cmd, cwd=cwd)
    if res.returncode != 0:
        raise SystemExit(res.returncode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        "-p",
        type=str,
        help="data folder containing protein subfolders",
        required=True
    )
    parser.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=1,
        help="number of parallel jobs to pass to scripts"
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default="auto",
        help="device to use for ESM embedding generation",
        choices=["auto", "cpu", "cuda"]
    )
    parser.add_argument(
        "--batch",
        "-b",
        type=int,
        default=1,
        help="batch size for ESM embedding generation"
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=4.0,
        help="distance threshold for binding site detection"
    )

    args = parser.parse_args()

    data_root = Path(args.data_dir)
    if not data_root.exists():
        raise SystemExit(f"Data root not found: {data_root}")

    gen_script = SCRIPTS / "generate_esm_embeddings.py"
    extract_binding_info_script = SCRIPTS / "extract_binding_info.py"

    for script in [gen_script, extract_binding_info_script]:
        if not script.exists():
            raise SystemExit(f"Script not found: {script}")

    if args.batch <= 0:
        raise SystemExit(f"Batch size must be a positive integer, got: {args.batch}")

    device = args.device if args.device != "auto" else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("\n" + "=" * 60)
    print(f"Processing dataset: {data_root}")
    print("=" * 60)

    print("Generating ESM embeddings...")
    esm_jobs = args.jobs if device != "cuda" else 1
    run_command([
        sys.executable,
        str(gen_script),
        "-p", str(data_root),
        "-j", str(esm_jobs),
        "-d", device,
        "-b", str(args.batch)
    ])

    print("Extracting binding info...")
    run_command([
        sys.executable,
        str(extract_binding_info_script),
        "-p", str(data_root),
        "-j", str(args.jobs),
        "-t", str(args.threshold),
        "-b", "processes"
    ])


if __name__ == '__main__':
    main()
