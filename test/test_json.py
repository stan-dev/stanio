import collections
import json
import os
import tempfile
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

from stanio.json import write_stan_json


@pytest.fixture(scope="module")
def TMPDIR():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


def compare_dictionaries(d1, d2):
    assert d1.keys() == d2.keys()
    for k in d1:
        data_1 = d1[k]
        data_2 = d2[k]
        if isinstance(data_1, dict):
            compare_dictionaries(data_1, data_2)
            continue

        if isinstance(data_2, collections.abc.Collection):
            data_2 = np.asarray(data_2).tolist()
        # np properly handles NaN equality
        np.testing.assert_equal(data_1, data_2)


def compare_before_after(file, dict, dict_exp):
    write_stan_json(file, dict)
    with open(file) as fd:
        dict_after = json.load(fd)
    compare_dictionaries(dict_after, dict_exp)
    return dict_after


def check_roundtrips(file, dict):
    compare_before_after(file, dict, dict)


def test_basic_list(TMPDIR) -> None:
    dict_list = {"a": [1.0, 2.0, 3.0]}
    file_list = os.path.join(TMPDIR, "list.json")
    check_roundtrips(file_list, dict_list)


def test_basic_array(TMPDIR) -> None:
    arr = np.repeat(3, 4)
    dict_vec = {"a": arr}
    file_vec = os.path.join(TMPDIR, "vec.json")
    check_roundtrips(file_vec, dict_vec)


def test_bool(TMPDIR) -> None:
    dict_bool = {"a": False, "b": True}
    file_bool = os.path.join(TMPDIR, "bool.json")
    dict_exp = {"a": 0, "b": 1}
    after = compare_before_after(file_bool, dict_bool, dict_exp)
    assert isinstance(after["a"], int)
    assert not isinstance(after["a"], bool)
    assert isinstance(after["b"], int)
    assert not isinstance(after["b"], bool)


def test_none(TMPDIR) -> None:
    dict_none = {"a": None}
    file_none = os.path.join(TMPDIR, "none.json")
    check_roundtrips(file_none, dict_none)


def test_pandas(TMPDIR) -> None:
    arr = np.repeat(3, 4)
    series = pd.Series(arr)
    dict_vec_pd = {"a": series}
    file_vec_pd = os.path.join(TMPDIR, "vec_pd.json")
    check_roundtrips(file_vec_pd, dict_vec_pd)

    dict_list = {"a": [1.0, 2.0, 3.0]}
    df_vec = pd.DataFrame(dict_list)
    file_pd = os.path.join(TMPDIR, "pd.json")
    compare_before_after(file_pd, df_vec, dict_list)


def test_empty(TMPDIR) -> None:
    dict_zero_vec: Dict[str, List[Any]] = {"a": []}
    file_zero_vec = os.path.join(TMPDIR, "empty_vec.json")
    check_roundtrips(file_zero_vec, dict_zero_vec)

    dict_zero_matrix: Dict[str, List[Any]] = {"a": [[], [], []]}
    file_zero_matrix = os.path.join(TMPDIR, "empty_matrix.json")
    check_roundtrips(file_zero_matrix, dict_zero_matrix)


def test_zero_shape(TMPDIR) -> None:
    arr = np.zeros(shape=(5, 0))
    dict_zero_matrix = {"a": arr}
    file_zero_matrix = os.path.join(TMPDIR, "empty_matrix.json")
    check_roundtrips(file_zero_matrix, dict_zero_matrix)


def test_3d(TMPDIR) -> None:
    arr = np.zeros(shape=(2, 3, 4))
    dict_3d_matrix = {"a": arr}
    file_3d_matrix = os.path.join(TMPDIR, "3d_matrix.json")
    check_roundtrips(file_3d_matrix, dict_3d_matrix)


def test_numpy_scalar(TMPDIR) -> None:
    scalr = np.int32(1)
    assert type(scalr).__module__ == "numpy"
    dict_scalr = {"a": scalr}
    file_scalr = os.path.join(TMPDIR, "scalr.json")
    check_roundtrips(file_scalr, dict_scalr)


def test_special_values(TMPDIR) -> None:
    # custom Stan serialization
    dict_inf_nan = {
        "a": np.array(
            [
                [-np.inf, np.inf, np.NaN],
                [-float("inf"), float("inf"), float("NaN")],
                [
                    np.float32(-np.inf),
                    np.float32(np.inf),
                    np.float32(np.NaN),
                ],
                [1e200 * -1e200, 1e220 * 1e200, -np.nan],
            ]
        )
    }
    dict_inf_nan_exp = {"a": [[-np.inf, np.inf, np.nan]] * 4}
    file_fin = os.path.join(TMPDIR, "inf.json")
    compare_before_after(file_fin, dict_inf_nan, dict_inf_nan_exp)


def test_complex_numbers(TMPDIR) -> None:
    dict_complex = {"a": np.array([np.complex64(3), 3 + 4j])}
    dict_complex_exp = {"a": [[3, 0], [3, 4]]}
    file_complex = os.path.join(TMPDIR, "complex.json")
    compare_before_after(file_complex, dict_complex, dict_complex_exp)


def test_tuples(TMPDIR) -> None:
    dict_tuples = {
        "a": (1, 2, 3),
        "b": [(1, [2, 3]), (4, [5, 6])],
        "c": ((1, np.array([1, 2.0, 3])), (3, np.array([1, 2, 3]))),
        "m": {"1": 1, "2": [2, 3]},
    }
    dict_tuple_exp = {
        "a": {"1": 1, "2": 2, "3": 3},
        "b": [{"1": 1, "2": [2, 3]}, {"1": 4, "2": [5, 6]}],
        "c": {"1": {"1": 1, "2": [1, 2.0, 3]}, "2": {"1": 3, "2": [1, 2, 3]}},
        "m": {"1": 1, "2": [2, 3]},
    }
    file_tuple = os.path.join(TMPDIR, "tuple.json")
    write_stan_json(file_tuple, dict_tuples)
    compare_before_after(file_tuple, dict_tuples, dict_tuple_exp)
