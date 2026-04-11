import ctypes
from ctypes import POINTER, c_bool, c_char_p, c_double, c_int, c_void_p
from enum import IntEnum
from importlib import metadata as importlib_metadata
from numbers import Integral
import os
from pathlib import Path

import numpy as np
import numpy.ctypeslib as npct

array_1d_double = npct.ndpointer(dtype=np.double, ndim=1, flags="CONTIGUOUS")
array_2d_double = npct.ndpointer(dtype=np.double, ndim=2, flags="CONTIGUOUS")
array_1d_int = npct.ndpointer(dtype=np.int32, ndim=1, flags="CONTIGUOUS")
array_2d_int = npct.ndpointer(dtype=np.int32, ndim=2, flags="CONTIGUOUS")
array_1d_uint16 = npct.ndpointer(dtype=np.uint16, ndim=1, flags="CONTIGUOUS")
array_2d_uint16 = npct.ndpointer(dtype=np.uint16, ndim=2, flags="CONTIGUOUS")

_LIBRARY_BASENAME = "_omnigbdt"
_LOADED_LIBRARIES = {}
_DLL_DIRECTORIES = []


class Verbosity(IntEnum):
    SILENT = 0
    SUMMARY = 1
    FULL = 2


def _normalize_verbosity(user_params=None, params=None):
    """Normalize the public verbosity parameter to an integer level.

    Args:
        user_params: Optional user-supplied parameter mapping.
        params: Optional merged parameter mapping.

    Returns:
        int: One of the integer ``Verbosity`` levels.

    Raises:
        ValueError: If the supplied verbosity value is unsupported.
    """
    user_params = {} if user_params is None else user_params
    params = {} if params is None else params

    value = user_params["verbosity"] if "verbosity" in user_params else params.get("verbose", True)

    if isinstance(value, str):
        normalized = value.strip().upper()
        mapping = {
            "SILENT": Verbosity.SILENT,
            "SUMMARY": Verbosity.SUMMARY,
            "FULL": Verbosity.FULL,
        }
        if normalized not in mapping:
            raise ValueError(
                "verbosity must be one of 'silent', 'summary', 'full', "
                "a Verbosity value, or an integer in {0, 1, 2}."
            )
        return int(mapping[normalized])

    if isinstance(value, bool):
        return int(Verbosity.FULL if value else Verbosity.SILENT)

    if isinstance(value, Verbosity):
        return int(value)

    if isinstance(value, Integral) and int(value) in (0, 1, 2):
        return int(value)

    raise ValueError(
        "verbosity must be one of Verbosity.SILENT, Verbosity.SUMMARY, "
        "Verbosity.FULL, or an integer in {0, 1, 2}."
    )


def _normalize_deterministic(user_params=None, params=None):
    """Normalize the public deterministic CPU mode parameter.

    Args:
        user_params: Optional user-supplied parameter mapping.
        params: Optional merged parameter mapping.

    Returns:
        bool: Normalized deterministic CPU mode flag.

    Raises:
        ValueError: If the supplied deterministic value is unsupported.
    """
    user_params = {} if user_params is None else user_params
    params = {} if params is None else params

    value = (
        user_params["deterministic"]
        if "deterministic" in user_params
        else params.get("deterministic", True)
    )

    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    if isinstance(value, Integral) and int(value) in (0, 1):
        return bool(value)

    raise ValueError("deterministic must be a bool or an integer in {0, 1}.")


def _candidate_library_names():
    if os.name == "nt":
        return [f"{_LIBRARY_BASENAME}.dll"]
    if os.name == "posix" and os.uname().sysname == "Darwin":
        return [f"{_LIBRARY_BASENAME}.dylib", f"{_LIBRARY_BASENAME}.so"]
    return [f"{_LIBRARY_BASENAME}.so"]


def _find_library_in_directory(directory):
    for name in _candidate_library_names():
        candidate = directory / name
        if candidate.is_file():
            return candidate.resolve()
    return None


def _resolve_installed_distribution_library_path(distribution_name="omnigbdt"):
    """Resolve the native library from an installed wheel distribution.

    This fallback is useful when tests run from a source checkout while the
    wheel under test is installed into site-packages. In that situation,
    ``omnigbdt.__file__`` points at the source tree, but the shared library
    still lives in the installed distribution.

    Args:
        distribution_name: Installed distribution name to inspect.

    Returns:
        Path | None: Resolved library path when found, otherwise ``None``.
    """
    try:
        distribution = importlib_metadata.distribution(distribution_name)
    except importlib_metadata.PackageNotFoundError:
        return None

    candidate_names = set(_candidate_library_names())
    for file_path in distribution.files or ():
        relative_path = Path(file_path)
        if relative_path.name not in candidate_names:
            continue
        resolved_path = Path(distribution.locate_file(relative_path)).resolve()
        if resolved_path.is_file():
            return resolved_path

    distribution_root = Path(distribution.locate_file("")).resolve()
    for directory in (distribution_root / "omnigbdt", distribution_root):
        direct_match = _find_library_in_directory(directory)
        if direct_match is not None:
            return direct_match

    return None


def _resolve_packaged_library_path():
    """Resolve the bundled native library path for the current environment.

    Returns:
        Path: Absolute path to the packaged native library.

    Raises:
        FileNotFoundError: If no packaged or installed native library can be
            found.
    """
    package_dir = Path(__file__).resolve().parent
    search_roots = [package_dir, package_dir.parent]
    for root in search_roots:
        direct_match = _find_library_in_directory(root)
        if direct_match is not None:
            return direct_match

    for build_dir_name in ("build", "_skbuild"):
        build_dir = package_dir.parent / build_dir_name
        if build_dir.is_dir():
            for name in _candidate_library_names():
                matches = sorted(build_dir.rglob(name))
                if matches:
                    return matches[0].resolve()

    installed_match = _resolve_installed_distribution_library_path()
    if installed_match is not None:
        return installed_match

    raise FileNotFoundError(
        "Could not find the packaged OmniGBDT native library. "
        "Install the package with pip or pass an explicit library path to load_lib()."
    )


def _resolve_library_path(path=None):
    if path is None:
        return _resolve_packaged_library_path()

    candidate = Path(path)
    if candidate.is_file():
        return candidate.resolve()
    if candidate.is_dir():
        library_path = _find_library_in_directory(candidate)
        if library_path is not None:
            return library_path

    raise FileNotFoundError(f"Could not find an OmniGBDT native library at {path!r}.")


def _configure_library(lib):
    """Attach ctypes signatures to the loaded native library.

    Args:
        lib: Loaded ``ctypes.CDLL`` instance for OmniGBDT.
    """
    lib.SetData.argtypes = [c_void_p, array_2d_uint16, array_2d_double, array_2d_double, c_int, c_bool]
    lib.SetBin.argtypes = [c_void_p, array_1d_uint16, array_1d_double]
    lib.SetGH.argtypes = [c_void_p, POINTER(c_double), POINTER(c_double)]
    lib.SetBaseScore.argtypes = [c_void_p, POINTER(c_double), c_int]
    lib.GetBaseScoreSize.argtypes = [c_void_p]
    lib.GetBaseScore.argtypes = [c_void_p, POINTER(c_double)]
    lib.SetData.restype = None
    lib.SetBin.restype = None
    lib.SetGH.restype = None
    lib.SetBaseScore.restype = None
    lib.GetBaseScoreSize.restype = c_int
    lib.GetBaseScore.restype = None

    lib.Boost.argtypes = [c_void_p]
    lib.Train.argtypes = [c_void_p, c_int]
    lib.TrimTrees.argtypes = [c_void_p, c_int]
    lib.Dump.argtypes = [c_void_p, c_char_p]
    lib.Load.argtypes = [c_void_p, c_char_p]
    lib.Boost.restype = None
    lib.Train.restype = None
    lib.TrimTrees.restype = None
    lib.Dump.restype = None
    lib.Load.restype = None

    lib.SetLabelDouble.restype = None
    lib.SetLabelInt.restype = None
    lib.Predict.restype = None

    lib.MultiNew.argtypes = [
        c_int,
        c_int,
        c_int,
        c_char_p,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        c_double,
        c_double,
        c_double,
        c_double,
        c_double,
        c_int,
        c_bool,
        c_int,
        c_int,
    ]
    lib.MultiNew.restype = c_void_p

    lib.SingleNew.argtypes = [
        c_int,
        c_char_p,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        c_double,
        c_double,
        c_double,
        c_double,
        c_double,
        c_int,
        c_int,
        c_int,
    ]
    lib.SingleNew.restype = c_void_p

    lib.TrainMulti.argtypes = [c_void_p, c_int, c_int]
    lib.PredictMulti.argtypes = [c_void_p, array_2d_double, array_1d_double, c_int, c_int, c_int]
    lib.Reset.argtypes = [c_void_p]
    lib.SingleFree.argtypes = [c_void_p]
    lib.MultiFree.argtypes = [c_void_p]
    lib.TrainMulti.restype = None
    lib.PredictMulti.restype = None
    lib.Reset.restype = None
    lib.SingleFree.restype = None
    lib.MultiFree.restype = None


def load_lib(path=None):
    """Load the OmniGBDT native library.

    Args:
        path: Optional explicit library path or containing directory.

    Returns:
        ctypes.CDLL: Configured native library handle.
    """
    library_path = _resolve_library_path(path)
    cache_key = str(library_path)
    if cache_key in _LOADED_LIBRARIES:
        return _LOADED_LIBRARIES[cache_key]

    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        _DLL_DIRECTORIES.append(os.add_dll_directory(str(library_path.parent)))

    lib = ctypes.CDLL(str(library_path))
    _configure_library(lib)
    _LOADED_LIBRARIES[cache_key] = lib
    return lib


def default_params():
    """Return the default Python parameter mapping for OmniGBDT.

    Returns:
        dict[str, object]: Default parameter values for booster construction.
    """
    return {
        "max_depth": 4,
        "max_leaves": 32,
        "max_bins": 128,
        "topk": 0,
        "deterministic": True,
        "seed": 0,
        "num_threads": 2,
        "min_samples": 20,
        "subsample": 1.0,
        "lr": 0.05,
        "base_score": None,
        "reg_l1": 0.0,
        "reg_l2": 1.0,
        "gamma": 1e-3,
        "loss": b"mse",
        "early_stop": 15,
        "one_side": True,
        "verbosity": int(Verbosity.FULL),
        "verbose": True,
        "hist_cache": 16,
    }


__all__ = [
    "array_1d_double",
    "array_2d_double",
    "array_1d_int",
    "array_2d_int",
    "array_1d_uint16",
    "array_2d_uint16",
    "default_params",
    "load_lib",
    "Verbosity",
]
