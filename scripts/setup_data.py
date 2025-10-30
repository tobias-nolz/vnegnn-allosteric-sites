import pandas as pd
from pathlib import Path

from prepare_pdb import prepare_pdb_directory
from prepare_ligand import prepare_ligands_from_asd
from tqdm import tqdm
import argparse


def setup_data(
        pdb_dir: Path,
        asd_dataset: pd.DataFrame,
        workers: int,
        skip_existing: bool = False,
) -> None:
    """
    Setup data for VN-EGNN allosteric site prediction.

    This will:
      - download PDB files into `pdb_dir` (one folder per PDB with `protein.pdb` inside)
      - extract ligand PDBs from the downloaded protein files

    :param pdb_dir: Path to the directory where PDB files will be stored.
    :param asd_dataset: DataFrame containing ASD dataset information.
    :param workers: Number of parallel workers for downloads/extraction.
    :param skip_existing: If True, skip creating files that already exist.
    :return: None
    """
    pdb_dir = Path(pdb_dir)

    if asd_dataset is None or asd_dataset.empty:
        raise ValueError("ASD dataset is empty or not provided")

    # Extract PDB ids from the ASD dataset
    if 'allosteric_pdb' not in asd_dataset.columns:
        raise KeyError("ASD dataset must contain column 'allosteric_pdb'")

    pdb_ids = asd_dataset['allosteric_pdb'].tolist()
    tqdm.write(f"Preparing PDB directory at {pdb_dir} with {len(pdb_ids)} PDBs (workers={workers})")
    prepare_pdb_directory(
        pdb_dir=pdb_dir,
        pdb_ids=pdb_ids,
        workers=workers
    )

    if not {'modulator_chain', 'modulator_resi'}.issubset(asd_dataset.columns):
        raise KeyError("ASD dataset must contain columns 'modulator_chain' and 'modulator_resi' for ligand extraction")

    ligand_info = pd.DataFrame(
        {
            'pdb_id': asd_dataset['allosteric_pdb'],
            'ligand_chain': asd_dataset['modulator_chain'],
            'ligand_residue': asd_dataset['modulator_resi'],
        }
    )
    tqdm.write(f"Extracting ligands (skip_existing={skip_existing}, workers={workers})")
    prepare_ligands_from_asd(
        pdb_dir=pdb_dir,
        ligand_info=ligand_info,
        skip_existing=skip_existing,
        workers=workers
    )


def _read_asd_dataset(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"ASD dataset file not found: {path}")
    return pd.read_csv(path, sep=None, engine='python')


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Setup VN-EGNN data: download PDBs and extract ligands from ASD dataset"
    )
    parser.add_argument(
        "--data-dir",
        "-ü",
        required=True,
        help="Directory where PDB folders will be stored (one folder per PDB)"
    )
    parser.add_argument(
        "--asd-file",
        "-a",
        required=True,
        help="Path to ASD dataset file (CSV/TSV) containing columns 'allosteric_pdb','modulator_chain','modulator_resi'"
    )
    parser.add_argument(
        "--workers",
        "-j",
        type=int,
        default=16,
        help="Number of parallel workers"
    )
    parser.add_argument(
        "--no-skip",
        action='store_true',
        help="Do not skip existing extracted ligand files"
    )

    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)
    asd_file = Path(args.asd_file)
    workers = args.workers
    skip_existing = not args.no_skip

    print(f"Loading ASD dataset from: {asd_file}")
    asd_df = _read_asd_dataset(asd_file)

    data_dir.mkdir(parents=True, exist_ok=True)

    setup_data(
        pdb_dir=data_dir,
        asd_dataset=asd_df,
        workers=workers,
        skip_existing=skip_existing
    )


if __name__ == '__main__':
    main()
