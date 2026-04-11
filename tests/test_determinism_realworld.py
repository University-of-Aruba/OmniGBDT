from __future__ import annotations

import os

import pytest

from .determinism_utils import (
    assert_repeated_runs_are_deterministic,
    make_realworld_determinism_dataset_from_mirror,
    train_multi_determinism_run,
    train_single_determinism_run,
)


if os.environ.get("OMNIGBDT_RUN_REALWORLD") != "1":
    pytestmark = pytest.mark.skip(reason="Real-world determinism benchmark is opt-in.")
else:
    pytest.importorskip("openpyxl")


def _load_realworld_dataset():
    """Load the fixed UCI mirror dataset split used for repeatability.

    Returns:
        DeterminismDataset: Fixed real-world determinism dataset splits.
    """
    try:
        return make_realworld_determinism_dataset_from_mirror()
    except Exception as exc:  # pragma: no cover - depends on external service.
        pytest.skip(f"Real-world determinism benchmark could not load the UCI mirror dataset: {exc}")


@pytest.mark.parametrize("num_threads", [2, 4])
@pytest.mark.parametrize("early_stop", [0, 15])
def test_singleoutput_realworld_runs_are_deterministic(
    tmp_path,
    num_threads,
    early_stop,
):
    """Ensure repeated real-world single-output runs stay identical.

    Args:
        tmp_path: Temporary directory for dumped models.
        num_threads: Requested native thread count from pytest parametrization.
        early_stop: Early-stopping patience from pytest parametrization.
    """
    dataset = _load_realworld_dataset()
    assert_repeated_runs_are_deterministic(
        lambda model_path: train_single_determinism_run(
            dataset=dataset,
            num_threads=num_threads,
            early_stop=early_stop,
            num_rounds=25,
            model_path=model_path,
        ),
        tmp_path,
    )


@pytest.mark.parametrize("num_threads", [2, 4])
@pytest.mark.parametrize("early_stop", [0, 15])
def test_multioutput_realworld_runs_are_deterministic(
    tmp_path,
    num_threads,
    early_stop,
):
    """Ensure repeated real-world multi-output runs stay identical.

    Args:
        tmp_path: Temporary directory for dumped models.
        num_threads: Requested native thread count from pytest parametrization.
        early_stop: Early-stopping patience from pytest parametrization.
    """
    dataset = _load_realworld_dataset()
    assert_repeated_runs_are_deterministic(
        lambda model_path: train_multi_determinism_run(
            dataset=dataset,
            num_threads=num_threads,
            early_stop=early_stop,
            num_rounds=25,
            model_path=model_path,
        ),
        tmp_path,
    )
