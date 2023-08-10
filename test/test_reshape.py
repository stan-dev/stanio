from cmath import rect
from pathlib import Path

import numpy as np
import pytest

from stanio.reshape_params import ParameterAccessor

HERE = Path(__file__).parent
DATA = HERE / "data"


# see file data/rectangles/output.stan
@pytest.fixture(scope="module")
def rect_data():
    files = [DATA / "rectangles" / f"output_{i}.csv" for i in range(1, 5)]
    accessor = ParameterAccessor.from_file(files)
    yield accessor.as_dict()


def test_basic_shapes(rect_data):
    assert rect_data["lp__"].shape == (1000, 4)
    assert rect_data["mu"].shape == (1000, 4)

    assert rect_data["v"].shape == (2, 1000, 4)
    assert rect_data["r"].shape == (3, 1000, 4)
    assert rect_data["m"].shape == (2, 3, 1000, 4)
    assert rect_data["threeD"].shape == (4, 2, 3, 1000, 4)

    assert rect_data["z"].shape == (1000, 4)
    assert rect_data["z"].dtype.kind == "c"
    assert rect_data["zv"].shape == (2, 1000, 4)
    assert rect_data["zv"].dtype.kind == "c"
    assert rect_data["zm"].shape == (2, 3, 1000, 4)
    assert rect_data["zm"].dtype.kind == "c"
    assert rect_data["z3D"].shape == (4, 2, 3, 1000, 4)
    assert rect_data["z3D"].dtype.kind == "c"


def check_rectangle(rect_data, draw, chain):
    # in output.stan we define all output variables as
    # multiples of mu, so we can check that the data is
    # reshaped correctly by checking that the multiples
    mu = rect_data["mu"][draw, chain]
    np.testing.assert_almost_equal(rect_data["v"][:, draw, chain], [mu, 2 * mu])
    np.testing.assert_almost_equal(
        rect_data["r"][:, draw, chain], [2 * mu, 3 * mu, 4 * mu]
    )
    matrix_expected = np.linspace(5, 11, 6).reshape(2, 3, order="F") * mu

    np.testing.assert_almost_equal(rect_data["m"][:, :, draw, chain], matrix_expected)
    threeD_expected = (
        np.linspace(5, 11, 6).reshape(1, 2, 3, order="F")
        * np.arange(1, 5).reshape(4, 1, 1)
        * mu
    )
    np.testing.assert_almost_equal(
        rect_data["threeD"][:, :, :, draw, chain],
        threeD_expected,
    )

    nu = rect_data["nu"][draw, chain]
    np.testing.assert_almost_equal(rect_data["z"][draw, chain], nu + 2j * nu)
    np.testing.assert_almost_equal(
        rect_data["zv"][:, draw, chain], [3 * nu + 4j * nu, 5 * nu + 6j * nu]
    )
    np.testing.assert_almost_equal(
        rect_data["zm"][:, :, draw, chain],
        matrix_expected + 1j * (matrix_expected + 1),
    )
    np.testing.assert_almost_equal(
        rect_data["z3D"][:, :, :, draw, chain],
        threeD_expected + 1j * (threeD_expected + 1),
    )


def test_basic_values(rect_data):
    # hardcoded to make sure we're really reading the chain and draw
    # we think we are
    assert rect_data["mu"][0, 0] == 0.5393776428
    assert rect_data["mu"][0, 1] == -0.8009042915

    check_rectangle(rect_data, 0, 0)
    check_rectangle(rect_data, -1, -1)
    for _ in range(100):
        draw = np.random.randint(1000)
        chain = np.random.randint(4)
        check_rectangle(rect_data, draw, chain)


# see file data/tuples/output.stan
@pytest.fixture(scope="module")
def tuple_data():
    files = [DATA / "tuples" / f"output_{i}.csv" for i in range(1, 5)]
    accessor = ParameterAccessor.from_file(files)
    yield accessor.as_dict()


def test_tuple_shapes(tuple_data):
    assert isinstance(tuple_data["pair"][0, 0], tuple)
    assert len(tuple_data["pair"][0, 0]) == 2

    assert isinstance(tuple_data["nested"][0, 0], tuple)
    assert len(tuple_data["nested"][0, 0]) == 2
    assert isinstance(tuple_data["nested"][0, 0][1], tuple)
    assert len(tuple_data["nested"][0, 0][1]) == 2

    assert tuple_data["arr_pair"].shape == (2, 1000, 4)
    assert isinstance(tuple_data["arr_pair"][0, 0, 0], tuple)

    assert tuple_data["arr_very_nested"].shape == (3, 1000, 4)
    assert isinstance(tuple_data["arr_very_nested"][0, 0, 0], tuple)
    assert isinstance(tuple_data["arr_very_nested"][0, 0, 0][0], tuple)
    assert isinstance(tuple_data["arr_very_nested"][0, 0, 0][0][1], tuple)

    assert tuple_data["arr_2d_pair"].shape == (3, 2, 1000, 4)
    assert isinstance(tuple_data["arr_2d_pair"][0, 0, 0, 0], tuple)

    assert tuple_data["ultimate"].shape == (2, 3, 1000, 4)
    assert isinstance(tuple_data["ultimate"][0, 0, 0, 0], tuple)
    assert tuple_data["ultimate"][0, 0, 0, 0][0].shape == (2,)
    assert isinstance(tuple_data["ultimate"][0, 0, 0, 0][0][0], tuple)
    assert tuple_data["ultimate"][0, 0, 0, 0][0][0][1].shape == (2,)
    assert tuple_data["ultimate"][0, 0, 0, 0][1].shape == (4, 5)


def assert_tuple_equal(t1, t2):
    assert len(t1) == len(t2)
    for x, y in zip(t1, t2):
        if isinstance(x, tuple):
            assert_tuple_equal(x, y)
        else:
            np.testing.assert_almost_equal(x, y)


def check_tuples(tuple_data, draw, chain):
    base = tuple_data["base"][draw, chain]
    base_i = tuple_data["base_i"][draw, chain]
    pair_exp = (base, 2 * base)
    np.testing.assert_almost_equal(tuple_data["pair"][draw, chain], pair_exp)
    nested_exp = (base * 3, (base_i, 4j * base))
    assert_tuple_equal(tuple_data["nested"][draw, chain], nested_exp)

    assert_tuple_equal(tuple_data["arr_pair"][0, draw, chain], pair_exp)
    assert_tuple_equal(tuple_data["arr_pair"][1, draw, chain], (base * 5, base * 6))

    assert_tuple_equal(
        tuple_data["arr_very_nested"][0, draw, chain], (nested_exp, base * 7)
    )
    assert_tuple_equal(
        tuple_data["arr_very_nested"][1, draw, chain],
        ((base * 8, (base_i * 2, base * 9.0j)), base * 10),
    )
    assert_tuple_equal(
        tuple_data["arr_very_nested"][2, draw, chain], (nested_exp, base * 11)
    )

    for i in range(3):
        for j in range(2):
            idx = i * 4 + j * 2 + 12
            assert_tuple_equal(
                tuple_data["arr_2d_pair"][i, j, draw, chain],
                (base * idx, base * (idx + 1)),
            )

    for i in range(2):
        for j in range(3):
            idx = i * 3 + j
            base_p = base + idx
            assert_tuple_equal(
                tuple_data["ultimate"][i, j, draw, chain][0][0],
                (base_p, [base_p * 2, base_p * 3]),
            )
            assert_tuple_equal(
                tuple_data["ultimate"][i, j, draw, chain][0][1],
                (base_p * 4, [base_p * 5, base_p * 6]),
            )
            matrix_expected = np.linspace(7, 11, 20).reshape(4, 5, order="F") * base_p

            np.testing.assert_almost_equal(
                tuple_data["ultimate"][i, j, draw, chain][1], matrix_expected
            )


def test_tuple_values(tuple_data):
    # fixed param
    assert (tuple_data["lp__"] == 0).all()

    # assert tuple_data["base"][0, 0] == 1.585448118

    check_tuples(tuple_data, 0, 0)
    check_tuples(tuple_data, -1, -1)
    for _ in range(100):
        draw = np.random.randint(1000)
        chain = np.random.randint(4)
        check_tuples(tuple_data, draw, chain)
