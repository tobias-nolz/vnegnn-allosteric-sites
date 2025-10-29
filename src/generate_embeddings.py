import os
import torch
from tqdm import tqdm
import sys
from pathlib import Path
from utils import all_exist_in_directory


def generate_embeddings(
        pdb_path: Path,
        vnegnn_path: Path,
        model: str = "esm2_t33_650M_UR50D",
        output_format: str = "npz",
        device: str = None,
        workers: int = 1,
        verbose: bool = True,
        batch_size: int = 1,
        monitor_memory: bool = True,
        check_if_exists: bool = True
) -> None:
    """
    Uses the VN-EGNN repository to generate ESM embeddings for protein structures in the specified PDB path.
    :param pdb_path: Path to the directory containing PDB files.
    :param vnegnn_path: Path to the VN-EGNN repository.
    :param model: ESM model to use for embedding generation.
    :param output_format: Format to save the embeddings ('npz' or 'h5').
    :param device: Device to run the model on (e.g., "cuda:0" or "cpu").
    :param workers: Number of parallel jobs to use.
    :param verbose: Whether to print verbose output.
    :param batch_size: Number of proteins to process in a batch.
    :param monitor_memory: Whether to monitor memory usage during embedding generation.
    :param check_if_exists: Whether to skip embedding generation if embeddings already exist.
    :return: None
    """
    if output_format not in ["npz", "hdf5"]:
        raise ValueError("output_format must be either 'npz' or 'hdf5'")

    if check_if_exists:
        filename = "embeddings.npz" if output_format == "npz" else "embeddings.hdf5"
        if all_exist_in_directory(directory=pdb_path, prefix=filename):
            tqdm.write("[INFO] All ESM embeddings already exist. Skipping generation.")
            return

    sys.path.append(os.path.join(vnegnn_path))
    from scripts.generate_esm_embeddings import generate_embeddings

    tqdm.write("[INFO] Generating ESM embeddings...")
    generate_embeddings.callback(
        path=pdb_path,
        model=model,
        output_format=output_format,
        device=device or "cuda:0" if torch.cuda.is_available() else "cpu",
        n_jobs=workers,
        verbose=verbose,
        batch_size=batch_size,
        monitor_memory=monitor_memory
    )
    tqdm.write("[INFO] ESM embeddings generation completed.")
