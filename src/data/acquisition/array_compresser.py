from pathlib import Path
from pickle import UnpicklingError
import logging

from tqdm import tqdm
import numpy as np


def load_compressed_array(file: str | Path) -> np.ndarray:
    """
    load the first numpy array in a compressed file.

    Parameters
    ----------
    file : str | Path
        path to the compressed npz file.

    Returns
    -------
    np.ndarray
        the loaded numpy array
    """
    npz_file = np.load(file)
    array_name = npz_file.files[0]

    return npz_file[array_name]


def save_compressed_array(file: str | Path, array: np.ndarray):
    """save a numpy array to a compressed file."""
    np.savez_compressed(file, array)


def compress_saved_array(file: Path, delete_uncompressed: bool = False):
    compressed_file = file.with_suffix(".npz")

    if not compressed_file.exists():
        array = np.load(file, allow_pickle=True)
        save_compressed_array(compressed_file, array)

    if delete_uncompressed:
        file.unlink()


def compress_dir(dir: Path, delete_unloadable: bool = True, delete_uncompressed: bool = True):
    """
    compress every .npy file in a directory.

    Parameters
    ----------
    dir : Path
        dir path
    delete_unloadable : bool, optional
        if `True`, delete the files that cannot be loaded by numpy, by default True
    delete_uncompressed : bool, optional
        if `True`, delete the uncompressed files once they have been compressed, by default True
    """
    files = list(dir.glob("*.npy"))

    unpickable = 0

    compression_progress = tqdm(files, total=len(files), maxinterval=1)
    for uncompressed_array in compression_progress:
        try:
            compress_saved_array(uncompressed_array, delete_uncompressed)
        except UnpicklingError:
            unpickable += 1

            if delete_unloadable:
                uncompressed_array.unlink()

    suffix_msg = ""
    if delete_uncompressed:
        suffix_msg = "and were deleted"

    logging.info("%s files weren't pickable %s.", unpickable, suffix_msg)
