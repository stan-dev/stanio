from dataclasses import dataclass
from enum import Enum
from math import prod
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import numpy.typing as npt


class VariableType(Enum):
    SCALAR = 1  # real or integer
    COMPLEX = 2  # complex number - requires striding
    TUPLE = 3  # tuples - require recursive handling


@dataclass
class Variable:
    # name of the parameter as given in stan. For nested parameters, this is a dummy name
    name: str
    # where to start (resp. end) reading from the flattened array.
    # For arrays with nested parameters, this will be for the first element
    # and is relative to the start of the parent
    start_idx: int
    end_idx: int
    # rectangular dimensions of the parameter (e.g. (2, 3) for a 2x3 matrix)
    # For nested parameters, this will be the dimensions of the outermost array.
    dimensions: Tuple[int, ...]
    # type of the parameter
    type: VariableType
    # list of nested parameters
    contents: List["Variable"]

    def columns(self) -> Iterable[int]:
        return range(self.start_idx, self.end_idx)

    def num_elts(self) -> int:
        return prod(self.dimensions)

    def elt_size(self) -> int:
        return self.end_idx - self.start_idx

    # total size is elt_size * num_elts

    def _extract_helper(self, src: np.ndarray, offset: int = 0):
        start = self.start_idx + offset
        end = self.end_idx + offset
        if self.type == VariableType.SCALAR:
            return src[..., start:end].reshape(-1, *self.dimensions, order="F")
        elif self.type == VariableType.COMPLEX:
            ret = src[..., start:end].reshape(-1, 2, *self.dimensions, order="F")
            ret = ret[:, ::2] + 1j * ret[:, 1::2]
            return ret.squeeze().reshape(-1, *self.dimensions, order="F")
        elif self.type == VariableType.TUPLE:
            out: np.ndarray = np.empty(
                (prod(src.shape[:-1]), prod(self.dimensions)), dtype=object
            )
            for idx in range(self.num_elts()):
                off = idx * self.elt_size() // self.num_elts()
                elts = [
                    param._extract_helper(src, offset=start + off)
                    for param in self.contents
                ]
                for i in range(elts[0].shape[0]):
                    out[i, idx] = tuple(elt[i] for elt in elts)
            return out.reshape(-1, *self.dimensions, order="F")

    def extract_reshape(self, src: np.ndarray) -> npt.NDArray[Any]:
        out = self._extract_helper(src)
        if src.ndim > 1:
            return out.reshape(*src.shape[:-1], *self.dimensions, order="F")
        else:
            return out.squeeze(axis=0)


def _munge_first_tuple(tup: str) -> str:
    return "dummy_" + tup.split(":", 1)[1]


def _get_base_name(param: str) -> str:
    return param.split(".")[0].split(":")[0]


def _from_header(header: str) -> List[Variable]:
    # appending __dummy ensures one extra iteration in the later loop
    header = header.strip() + ",__dummy"
    entries = header.split(",")
    params = []
    start_idx = 0
    name = _get_base_name(entries[0])
    for i in range(0, len(entries) - 1):
        entry = entries[i]
        next_name = _get_base_name(entries[i + 1])

        if next_name != name:
            if ":" not in entry:
                dims = entry.split(".")[1:]
                if ".real" in entry or ".imag" in entry:
                    type = VariableType.COMPLEX
                    dims = dims[:-1]
                else:
                    type = VariableType.SCALAR
                params.append(
                    Variable(
                        name=name,
                        start_idx=start_idx,
                        end_idx=i + 1,
                        dimensions=tuple(map(int, dims)),
                        type=type,
                        contents=[],
                    )
                )
            else:
                dims = entry.split(":")[0].split(".")[1:]
                munged_header = ",".join(
                    dict.fromkeys(map(_munge_first_tuple, entries[start_idx : i + 1]))
                )

                params.append(
                    Variable(
                        name=name,
                        start_idx=start_idx,
                        end_idx=i + 1,
                        dimensions=tuple(map(int, dims)),
                        type=VariableType.TUPLE,
                        contents=_from_header(munged_header),
                    )
                )

            start_idx = i + 1
            name = next_name

    return params


def parse_header(header: str) -> Dict[str, Variable]:
    return {param.name: param for param in _from_header(header)}


def stan_variables(
    parameters: Dict[str, Variable], source: npt.NDArray[np.float64]
) -> Dict[str, npt.NDArray[Any]]:
    return {param.name: param.extract_reshape(source) for param in parameters.values()}
