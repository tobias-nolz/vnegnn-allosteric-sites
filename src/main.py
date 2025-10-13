import os
import pandas as pd

from process_pdb import prepare_pdb_directory


if __name__ == "__main__":
    DATA_DIR = os.path.join("..", "data")
    PDB_DIR = os.path.join(DATA_DIR, "PDB")
    ASD_DIR = os.path.join(DATA_DIR, "ASD dataset")
    ASD_FILE = os.path.join(ASD_DIR, "ASD_Release_202309_AS.txt")
    PRED_DIR = os.path.join(DATA_DIR, "predictions")

    asd_dataset = pd.read_csv(ASD_FILE, sep="\t")
    pdb_ids = asd_dataset['allosteric_pdb'].tolist()
    prepare_pdb_directory(pdb_dir=PDB_DIR, pdb_ids=pdb_ids, workers=16)


