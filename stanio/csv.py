"""
Module to load the minimal information from a Stan CSV file.
Only the header row and data are read, no metadata is parsed.
"""

from typing import List, Tuple, Union
import numpy as np


def read_csv(filenames: Union[str, List[str]]) -> Tuple[str, np.ndarray]:
    if not isinstance(filenames, list):
        filenames = [filenames]

    header = ""
    data = [None for _ in range(len(filenames))]
    for i, f in enumerate(filenames):
        with open(f, "r") as fd:
            while (file_header := fd.readline()).startswith("#"):
                pass
            if header == "":
                header = file_header
            else:
                assert header == file_header, "Headers do not match"
            data[i] = np.loadtxt(fd, delimiter=",", comments="#")

    return header.strip(), np.stack(data, axis=0).T
