# batram

This repository contains a Python implementation of the method described in
[Katzfuss and Schäfer (2023)](https://doi.org/10.1080/01621459.2023.2197158) as
well as code from related research projects. We aim to provide a comprehensive
package integrating all developments. Please refer to the project specific
branches for now:

- [Wiemann and Katzfuss (2023)](https://link.springer.com/article/10.1007/s13253-023-00580-z): See branch
  [mvtm](https://github.com/katzfuss-group/batram/tree/mvtm)
- [Chakraborty and Katzfuss (2024)](https://arxiv.org/pdf/2409.19208): See branch [ShrinkTM](https://github.com/katzfuss-group/batram/tree/ShrinkTM)

## Installation

Run the following command in your local virtual environment. Please make sure
that a c++ compiler and the python headers are installed since this is needed to
install the dependecy [veccs](https://github.com/katzfuss-group/veccs).

1. `pip install -e .`

### Remarks

- We have observed that the software seems to be unstable on a MacBook Pro with
  M1 Pro chip when using conda instead of a regular python installation.


## How to contribute?

1. install the package with the additional dependencies for development using
   `pip install -e .[dev,docs]`
2. before pushing on `main` or a PR, run `pre-commit run --all-files` and ensure
   that all tests pass by running `pytest`.
3. before pushing on `main` or merging a PR, make sure the code is well
   documented and covered by tests.

The documentenation can be viewed while editing the code using `mkdocs serve`.

## Acknowledgements

An initial Python
[implementation](https://github.com/katzfuss-group/BaTraMaSpa_py) was provided
by [Jian Cao](https://www.uh.edu/nsm/math/people/faculty/index.php#assistantprof).
