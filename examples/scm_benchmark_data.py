"""Load the Mulan SCM20D product-price benchmark from its two publisher ZIPs."""

import hashlib
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np


SCM_ARCHIVE_SHA256 = {
    "scm20d-train.zip": "78fdf75a55a242b1ade1295ec04e962908790d6da020c9ba836f9961c264f336",
    "scm20d-test.zip": "e81aebcddf504e7457c595ee38d36c0d2b35eb2cb4506c960499104a83517d6a",
}


def _read_archive(path: Path) -> np.ndarray:
    """Read the verified numeric ARFF table without extracting files.

    Args:
        path: Path to the original SCM20D training or test ZIP.

    Returns:
        A float64 matrix with 61 features followed by 16 product prices.

    Raises:
        ValueError: If the ZIP checksum differs from the reference snapshot.
    """
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != SCM_ARCHIVE_SHA256[path.name]:
        raise ValueError(f"Checksum mismatch for {path.name}.")
    with ZipFile(BytesIO(data)) as archive:
        raw = archive.read(f"{path.stem}.arff")
    records = raw.split(b"@data", maxsplit=1)[1].strip()
    return np.loadtxt(BytesIO(records), delimiter=",")


def load_scm_split(directory: Path) -> tuple[np.ndarray, ...]:
    """Split the publisher's training data by game and use its full test set.

    Args:
        directory: Directory containing scm20d-train.zip and scm20d-test.zip.

    Returns:
        Contiguous feature and target arrays for training, validation and test,
        in that order. The three sets have 5,971, 1,492 and 1,503 rows.

    Raises:
        ValueError: If an archive differs from the reference snapshot.
    """
    train_pool = _read_archive(directory / "scm20d-train.zip")
    test = _read_archive(directory / "scm20d-test.zip")
    # A reset of the day counter identifies the start of the next game.
    game_starts = 1 + np.flatnonzero(np.diff(train_pool[:, 0]) < 0)
    first_validation_row = game_starts[31]  # First row of game 33 (of 40).
    train, valid = np.split(train_pool, [first_validation_row])
    return tuple(
        np.ascontiguousarray(columns)
        for rows in (train, valid, test)
        for columns in (rows[:, :61], rows[:, 61:])
    )
