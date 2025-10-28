import os
import pandas as pd
from pathlib import Path

from prepare_pdb import prepare_pdb_directory
from prepare_ligand import prepare_ligands_from_asd

from process_pdb import prepare_pdb_directory

def setup_data(pdb_dir: Path, asd_dataset: pd.DataFrame, vnegnn_path: Path):
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
        check_existing=False,
        workers=16
    )


if __name__ == "__main__":
    DATA_DIR = os.path.join("..", "data")
    PDB_DIR = Path(os.path.join(DATA_DIR, "PDB"))
    ASD_DIR = os.path.join(DATA_DIR, "ASD dataset")
    ASD_FILE = os.path.join(ASD_DIR, "ASD_Release_202309_AS.txt")
    PRED_DIR = os.path.join(DATA_DIR, "predictions")
    VNEGNN_PATH = Path(os.path.abspath(os.path.join("..", "..", "vnegnn")))

    asd_dataset = pd.read_csv(ASD_FILE, sep="\t")

    # Setup the data directory with PDB files and ligands
    setup_data(pdb_dir=PDB_DIR, asd_dataset=asd_dataset.iloc[:5], vnegnn_path=VNEGNN_PATH)

