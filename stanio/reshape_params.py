from math import prod
from attr import dataclass
from enum import Enum
import numpy as np

from .csv import read_csv


class ParameterType(Enum):
    SCALAR = 1  # real or integer
    COMPLEX = 2  # complex number - requires striding
    TUPLE = 3  # tuples - require recursive handling


@dataclass
class Parameter:
    # name of the parameter as given in stan. For nested parameters, this is a dummy name
    name: str
    # where to start (resp. end) reading from the flattened array.
    # For arrays with nested parameters, this will be for the first element.
    start_idx: int
    end_idx: int
    # rectangular dimensions of the parameter (e.g. (2, 3) for a 2x3 matrix)
    # For nested parameters, this will be the dimensions of the outermost array.
    dimensions: tuple[int]
    # type of the parameter
    type: ParameterType
    # list of nested parameters
    contents: list["Parameter"]

    def num_elts(self):
        return prod(self.dimensions)

    def elt_size(self):
        return self.end_idx - self.start_idx

    # total size is elt_size * num_elts

    def do_reshape(
        self, src: np.ndarray, *, offset: int = 0, original_shape: bool = True
    ):
        if original_shape:
            dims = src.shape[1:]
        else:
            dims = (-1,)
        start = self.start_idx + offset
        end = self.end_idx + offset
        if self.type == ParameterType.SCALAR:
            return src[start:end].reshape(*self.dimensions, *dims, order="F")
        elif self.type == ParameterType.COMPLEX:
            ret = src[start:end].reshape(2, *self.dimensions, -1, order="F")
            ret = ret[::2, ...] + 1j * ret[1::2, ...]
            return ret.squeeze().reshape(*self.dimensions, *dims, order="F")
        elif self.type == ParameterType.TUPLE:
            out = np.empty((prod(self.dimensions), prod(src.shape[1:])), dtype=object)
            for idx in range(self.num_elts()):
                off = idx * self.elt_size() // self.num_elts()
                elts = [
                    param.do_reshape(src, offset=off + offset, original_shape=False)
                    for param in self.contents
                ]
                for i in range(elts[-1].shape[-1]):
                    out[idx, i] = tuple(
                        # extra work to avoid scalar arrays
                        e.item() if (e := elt[..., i]).shape == () else e
                        for elt in elts
                    )
            return out.reshape(*self.dimensions, *dims, order="F")


def _munge_first_tuple(tup: str) -> str:
    return "dummy_" + tup.split(":", 1)[1]


def _get_base_name(param: str) -> str:
    return param.split(".")[0].split(":")[0]


# TODO: get rid of 'base' argument?
def _from_header(header: str, base: int = 0) -> list[Parameter]:
    header = header.strip() + ",__dummy"
    entries = header.split(",")
    params = []
    start_idx = base
    name = _get_base_name(entries[0])
    for i in range(0, len(entries) - 1):
        entry = entries[i]
        next_name = _get_base_name(entries[i + 1])

        if next_name != name:
            if ":" not in entry:
                dims = entry.split(".")[1:]
                if ".real" in entry or ".imag" in entry:
                    type = ParameterType.COMPLEX
                    dims = dims[:-1]
                else:
                    type = ParameterType.SCALAR
                params.append(
                    Parameter(
                        name=name,
                        start_idx=start_idx,
                        end_idx=base + i + 1,
                        dimensions=tuple(map(int, dims)),
                        type=type,
                        contents=[],
                    )
                )
            else:
                dims = entry.split(":")[0].split(".")[1:]
                munged_header = ",".join(
                    dict.fromkeys(
                        map(_munge_first_tuple, entries[start_idx - base : i + 1])
                    )
                )

                params.append(
                    Parameter(
                        name=name,
                        start_idx=start_idx,
                        end_idx=base + i + 1,
                        dimensions=tuple(map(int, dims)),
                        type=ParameterType.TUPLE,
                        contents=_from_header(munged_header, base=start_idx),
                    )
                )

            start_idx = base + i + 1
            name = next_name

    return params


class ParameterAccessor:
    def __init__(self, params: dict[Parameter], data: np.ndarray):
        self.params = params
        self._data = data
        # TODO: consider caching the reshaped data

    @classmethod
    def from_header(cls, header: str, data: np.ndarray) -> "ParameterAccessor":
        params = {param.name: param for param in _from_header(header)}
        return cls(params, data)

    @classmethod
    def from_file(cls, filename: str) -> "ParameterAccessor":
        header, data = read_csv(filename)
        return cls.from_header(header, data)

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            param.name: param.do_reshape(self._data) for param in self.params.values()
        }

    def __getitem__(self, key: str) -> np.ndarray:
        return self.params[key].do_reshape(self._data)

    def data(self) -> np.ndarray:
        return self._data
