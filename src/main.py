import os
import pandas as pd
from pathlib import Path

from prepare_pdb import prepare_pdb_directory
from prepare_ligand import prepare_ligands_from_asd
from generate_embeddings import generate_embeddings
from extract_binding_info import extract_binding_info


def setup_data(
        pdb_dir: Path,
        asd_dataset: pd.DataFrame,
        vnegnn_path: Path,
        msms_path: Path = None
) -> None:
    """
    Setup data for VN-EGNN allosteric site prediction.
    :param pdb_dir: Path to the directory where PDB files will be stored.
    :param asd_dataset: DataFrame containing ASD dataset information.
    :param vnegnn_path: Path to the VN-EGNN repository.
    :param msms_path: Path to the MSMS executable. Needed if not in PATH.
    :return: None
    """
    pdb_ids = asd_dataset['allosteric_pdb'].tolist()
    prepare_pdb_directory(pdb_dir=PDB_DIR, pdb_ids=pdb_ids, workers=16)

    ligand_info = pd.DataFrame(
        {
            'pdb_id': asd_dataset['allosteric_pdb'],
            'ligand_chain': asd_dataset['modulator_chain'],
            'ligand_residue': asd_dataset['modulator_resi'],
        }
    )
    prepare_ligands_from_asd(
        pdb_dir=pdb_dir,
        ligand_info=ligand_info,
        skip_existing=True,
        workers=16
    )

    generate_embeddings(
        pdb_path=PDB_DIR,
        vnegnn_path=vnegnn_path,
        workers=4,
        batch_size=8,
        monitor_memory=True,
        verbose=True,
        output_format="npz",
        check_if_exists=True
    )

    extract_binding_info(
        pdb_path=PDB_DIR,
        vnegnn_path=VNEGNN_PATH,
        threshold=4.0,
        workers=4,
        verbose=True,
        backend="processes",
        msms_path=msms_path,
        check_if_exists=True
    )


if __name__ == "__main__":
    DATA_DIR = Path(os.path.join("..", "data"))
    PDB_DIR = DATA_DIR / "PDB"
    PRED_DIR = DATA_DIR / "predictions"

    ASD_DIR = DATA_DIR / "ASD dataset"
    ASD_FILE = ASD_DIR / "ASD_Release_202309_AS.txt"

    VNEGNN_PATH = ...
    MSMS_PATH = ...

    asd_dataset = pd.read_csv(ASD_FILE, sep="\t")
    asd_dataset = asd_dataset.sample(n=8, random_state=42).reset_index(drop=True)

    setup_data(
        pdb_dir=PDB_DIR,
        asd_dataset=asd_dataset.iloc[:32],
        vnegnn_path=VNEGNN_PATH,
        msms_path=MSMS_PATH
    )

