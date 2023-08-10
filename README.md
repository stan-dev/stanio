# StanIO

A set of functions for data-wrangling in the formats used by
the [Stan](https://mc-stan.org) probabalistic programming language.

## Features

- Writing Python dictionaries to Stan-compatible JSON
- Basic reading of StanCSV files into numpy arrays
- Parameter extraction from numpy arrays based on StanCSV headers
  - e.g., if you have a `matrix[2,3] x` in a Stan program, extracting `x` will give you a `(num_draws, 2, 3)` numpy array
