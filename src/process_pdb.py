from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os
import re
import time
import requests


def normalize_pdb_tokens(pbd_id: str) -> list[str]:
    """
    Normalize and extract PDB IDs from a raw string.
    Handles cases like:
    - Single ID: "1ABC"
    - Multiple IDs: "1ABC; 2DEF"
    - IDs with chain: "1ABC_A"
    :param cell: cell value from the PDB ID column
    :return: normalized PDB IDs
    """
    s = pbd_id.strip()

    # split by common separators first
    parts = re.split(r'[;,/|\s]+', s)
    ids = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # look for the first alphanumeric run of length=4 (PDB ids are 4 chars)
        m = re.search(r'([A-Za-z0-9]{4})', p)
        if m:
            ids.append(m.group(1).upper())
    return ids


def download_single(
        pdb_id: str,
        out_dir: str,
        max_retries=3,
        timeout=25) -> tuple[str, str, str | None]:
    """
    Download a single PDB ID and save to out_dir.
    If the file already exists, skip download.
    :param pdb_id: PDB ID to download
    :param out_dir: Directory to save PDB files
    :param max_retries: Maximum number of retries on failure
    :param timeout: Request timeout in seconds
    :return: (pdb_id, status, file_path or None)
        status: "downloaded", "exists", "not_found"
        file_path: path to saved file or None if not downloaded
    """
    pdb_id_up = pdb_id.upper()
    out_pdb = os.path.join(out_dir, f"{pdb_id_up}.pdb")

    if os.path.exists(out_pdb):
        return pdb_id_up, "exists", None

    url = f"https://files.rcsb.org/download/{pdb_id_up}.pdb"
    attempt = 0

    while attempt < max_retries:
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code == 200 and r.content and len(r.content) > 100:
                with open(out_dir, "wb") as fh:
                    fh.write(r.content)
                return pdb_id_up, "downloaded", out_dir
            else:
                # 404 or small content -> return not found
                return pdb_id_up, "not_found", None
        except requests.RequestException as e:
            attempt += 1
            time.sleep(1 + attempt * 0.5)

    return pdb_id_up, "not_found", None


def prepare_pdb_directory(
        pdb_dir: str,
        pdb_ids: list[str],
        clear_existing: bool = False,
        workers: int = 8,
        print_summary: str = True) -> None:
    """
    Prepare a directory with PDB files for the given PDB IDs.
    Downloads files in parallel using multiple threads.
    :param pdb_dir: Directory to save PDB files
    :param pdb_ids: List of raw PDB ID strings (may contain multiple IDs per entry)
    :param clear_existing: If True, clear existing PDB files in the directory before downloading
    :param workers: Number of parallel download threads
    :param print_summary: If True, print a summary of download results
    :return: None
    """
    pdb_ids = sorted({pid for raw in pdb_ids if raw for pid in normalize_pdb_tokens(raw)})
    os.makedirs(pdb_dir, exist_ok=True)

    if clear_existing:
        existing_files = os.listdir(pdb_dir)
        for f in existing_files:
            if f.endswith(".pdb"):
                os.remove(os.path.join(pdb_dir, f))

    results = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(download_single, pid, pdb_dir): pid for pid in pdb_ids}
        iterator = as_completed(futures)
        if print_summary:
            iterator = tqdm(iterator, total=len(futures), desc="Downloading", unit="pdb")
        for fut in iterator:
            pid = futures[fut]
            try:
                res = fut.result()
                results.append(res)
                status = res[1]
                if status == "not_found":
                    print(f"[MISS] {pid} not found on RCSB")
            except Exception as e:
                print(f"[ERR] {pid} -> {e}")

    counts = {"downloaded": 0, "exists": 0, "not_found": 0}
    for _id, status, _path in results:
        counts[status] = counts.get(status, 0) + 1
    print("=== PDB processing summary ===")
    print(f"Downloaded: {counts.get('downloaded', 0)}")
    print(f"Already existed: {counts.get('exists', 0)}")
    print(f"Not found (404/empty): {counts.get('not_found', 0)}")
    print(f"Saved to: {os.path.abspath(pdb_dir)}")
    print("==============================")
    return None
