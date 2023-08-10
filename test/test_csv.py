from pathlib import Path

import numpy as np
import pytest

from stanio.csv import read_csv

HERE = Path(__file__).parent
DATA = HERE / "data"


def test_single_file() -> None:
    header, data = read_csv(str(DATA / "bernoulli" / "output_1.csv"))
    assert (
        header
        == "lp__,accept_stat__,stepsize__,treedepth__,n_leapfrog__,divergent__,energy__,theta"
    )
    assert data.shape == (1, 1000, 8)
    assert 0.2 < np.mean(data, axis=(0, 1))[7] < 0.3


def test_multiple_files() -> None:
    files = [str(DATA / "bernoulli" / f"output_{i}.csv") for i in range(1, 5)]
    header, data = read_csv(files)
    assert (
        header
        == "lp__,accept_stat__,stepsize__,treedepth__,n_leapfrog__,divergent__,energy__,theta"
    )
    assert data.shape == (4, 1000, 8)
    assert 0.2 < np.mean(data, axis=(0, 1))[7] < 0.3


def test_mismatched_files() -> None:
    files = [str(DATA / "bernoulli" / f"output_{i}.csv") for i in range(1, 5)]
    files[1] = str(DATA / "bernoulli" / "output_more_variables.csv")
    with pytest.raises(AssertionError, match="Headers do not match"):
        read_csv(files)

    files[1] = str(DATA / "bernoulli" / "output_missing_columns.csv")
    with pytest.raises(ValueError, match="must have the same shape"):
        read_csv(files)
