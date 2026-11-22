"""Load UCI Energy Efficiency and reproduce the observation and geometry holdouts."""

import hashlib
from io import BytesIO
from pathlib import Path

import numpy as np


ENERGY_CSV_SHA256 = "db44dbe453acd464b5cf65be2fb01a28aa9c5b2630300e65fbe28cde35f5d96f"


def load_energy_csv(path: Path):
    """Read the eight simulation inputs and the original heating and cooling loads.

    Args:
        path: UCI 242 CSV downloaded from the Machine Learning Repository.

    Returns:
        Feature matrix (768, 8), target matrix (768, 2) and one geometry ID
        per row. The first five features identify each building geometry.

    Raises:
        ValueError: If the CSV differs from the benchmark's reference file.
    """
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != ENERGY_CSV_SHA256:
        raise ValueError("Energy CSV does not match the UCI snapshot used by this benchmark.")
    data = np.loadtxt(BytesIO(content), delimiter=",", skiprows=1)
    geometry_ids = np.unique(data[:, :5], axis=0, return_inverse=True)[1].reshape(-1)
    return data[:, :8], data[:, 8:], geometry_ids


def energy_split(case: str, seed: int, energy):
    """Split individual observations or keep entire building geometries together.

    Args:
        case: Either "energy-rows" or "energy-geometries".
        seed: NumPy partition seed. The benchmark runs seeds 0 through 9.
        energy: Feature, target and geometry arrays from load_energy_csv.

    Returns:
        Training, validation and test feature/target pairs as six arrays.

    Raises:
        ValueError: If the requested split is unknown.
    """
    x, y, geometry_ids = energy
    rng = np.random.default_rng(seed)
    if case == "energy-rows":
        train, valid, test = np.split(rng.permutation(len(x)), [460, 614])
    elif case == "energy-geometries":
        groups = np.split(rng.permutation(np.unique(geometry_ids)), [7, 9])
        train, valid, test = [np.flatnonzero(np.isin(geometry_ids, group)) for group in groups]
    else:
        raise ValueError(f"Unknown Energy Efficiency split: {case}")
    return x[train], y[train], x[valid], y[valid], x[test], y[test]
