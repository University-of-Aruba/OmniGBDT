from __future__ import annotations

import pytest

from .determinism_utils import (
    assert_repeated_runs_are_deterministic,
    make_synthetic_determinism_dataset,
    train_multi_determinism_run,
    train_single_determinism_run,
)


@pytest.mark.parametrize("num_threads", [2, 4])
@pytest.mark.parametrize("early_stop", [0, 15])
def test_singleoutput_synthetic_runs_are_deterministic(
    tmp_path,
    num_threads,
    early_stop,
):
    """Ensure repeated synthetic single-output runs stay identical.

    Args:
        tmp_path: Temporary directory for dumped models.
        num_threads: Requested native thread count from pytest parametrization.
        early_stop: Early-stopping patience from pytest parametrization.
    """
    dataset = make_synthetic_determinism_dataset()
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
def test_multioutput_synthetic_runs_are_deterministic(
    tmp_path,
    num_threads,
    early_stop,
):
    """Ensure repeated synthetic multi-output runs stay identical.

    Args:
        tmp_path: Temporary directory for dumped models.
        num_threads: Requested native thread count from pytest parametrization.
        early_stop: Early-stopping patience from pytest parametrization.
    """
    dataset = make_synthetic_determinism_dataset()
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
