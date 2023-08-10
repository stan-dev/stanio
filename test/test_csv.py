from pathlib import Path

import numpy as np
import pytest

from stanio.csv import read_csv

HERE = Path(__file__).parent
DATA = HERE / "data"


def test_single_file():
    header, data = read_csv(DATA / "bernoulli" / "output_1.csv")
    assert (
        header
        == "lp__,accept_stat__,stepsize__,treedepth__,n_leapfrog__,divergent__,energy__,theta"
    )
    assert data.shape == (8, 1000, 1)
    assert 0.2 < np.mean(data, axis=(1, 2))[7] < 0.3


def test_multiple_files():
    files = [DATA / "bernoulli" / f"output_{i}.csv" for i in range(1, 5)]
    header, data = read_csv(files)
    assert (
        header
        == "lp__,accept_stat__,stepsize__,treedepth__,n_leapfrog__,divergent__,energy__,theta"
    )
    assert data.shape == (8, 1000, 4)
    assert 0.2 < np.mean(data, axis=(1, 2))[7] < 0.3


def test_mismatched_files():
    files = [DATA / "bernoulli" / f"output_{i}.csv" for i in range(1, 5)]
    files[1] = DATA / "bernoulli" / "output_more_variables.csv"
    with pytest.raises(AssertionError, match="Headers do not match"):
        read_csv(files)

    files[1] = DATA / "bernoulli" / "output_missing_columns.csv"
    with pytest.raises(ValueError, match="must have the same shape"):
        read_csv(files)
