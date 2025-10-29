import os
from tqdm import tqdm
import sys
from pathlib import Path
from utils import all_exist_in_directory


def extract_binding_info(
        pdb_path: Path,
        vnegnn_path: Path,
        threshold: float = 4.0,
        workers: int = 1,
        verbose: bool = True,
        backend: str = "processes",
        msms_path: Path = None,
        check_if_exists: bool = True
) -> None:
    """
    Uses the VN-EGNN repository to extract binding site information from protein structures in the specified PDB path.
    :param pdb_path: Path to the directory containing PDB files.
    :param vnegnn_path: Path to the VN-EGNN repository.
    :param threshold: Distance threshold to consider for binding site extraction.
    :param workers: Number of parallel jobs to use.
    :param verbose: Whether to print verbose output.
    :param backend: Backend to use for parallel processing ('threads' or 'processes').
    :param msms_path: Path to the MSMS executable, if not in system PATH.
    :param check_if_exists: Whether to skip extraction if binding info already exists.
    :return: None
    """
    sys.path.append(os.path.join(vnegnn_path))
    from scripts.extract_binding_info import extract_binding_info

    if check_if_exists:
        if all_exist_in_directory(directory=pdb_path, prefix="binding.npz"):
            tqdm.write("[INFO] All binding site information already exists. Skipping extraction.")
            return

    if msms_path is not None:
        os.environ["PATH"] += os.pathsep + str(msms_path)
        tqdm.write(f"[INFO] Added MSMS path to environment: {msms_path}")

    tqdm.write("[INFO] Extracting binding site information...")
    extract_binding_info.callback(
        path=pdb_path,
        n_jobs=workers,
        threshold=threshold,
        verbose=verbose,
        backend=backend
    )
    tqdm.write("[INFO] Binding site information extraction completed.")
