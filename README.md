# batram

Bayesian Transport Maps for non-Gaussian spatial fields. The package contains a python implementation of the method described in [Katzfuss and Schäfer (2023)](https://doi.org/10.1080/01621459.2023.2197158).

## Installation

Run the following command in your local virtual environment. Please make sure
that a c++ compiler and the python headers are installed since this is needed to
install the dependecy [veccs](https://github.com/katzfuss-group/veccs).

1. `pip install -e .`


## How to contribute?

1. install the package with the additional dependencies for development using
   `pip install -e .[dev,docs]`
2. before pushing on `main` or a PR, run `pre-commit run --all-files` and
   `pytest`.
3. before pushing on `main` or merging a PR, make sure the code is well
   documented and covered by tests.

The documentenation can be viewed while editing the code using `mkdocs serve`.

## Acknowledgements

An initial Python
[implementation](https://github.com/katzfuss-group/BaTraMaSpa_py) was provided
by Jian Cao. This work was supported by XXX [grant numbers xxxx, xxxx], YYY
[grant number yyyy]; and ZZZZ [grant number zzzz].

## A logo

- BatRam

  ![Image from stable diffusion](https://user-images.githubusercontent.com/603509/228377927-bbdf6cde-80cf-455b-8633-b7638e1b0327.png)
