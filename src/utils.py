import os
from pathlib import Path


def all_exist_in_directory(
        directory: Path,
        prefix: str
) -> bool:
    """
    Checks if in every subdirectory of the given directory, there exists at least one file with the given prefix.
    :param directory: Directory to check for files.
    :param prefix: Prefix of the files to look for.
    :return: True if files with all prefixes exist, False otherwise.
    """
    for root, dirs, files in os.walk(directory):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            if not any(fname.startswith(prefix) for fname in os.listdir(dir_path)):
                return False
    return True
