# batram

This repository hosts a Python implementation of the methodology presented in
[Katzfuss and Schäfer (2023)](https://doi.org/10.1080/01621459.2023.2197158),
along with code from related research projects. For project-specific
implementations, please refer to the corresponding branches.

- [Katzfuss and Schäfer (2023)](https://doi.org/10.1080/01621459.2023.2197158): An implemplementation of the foundational methodology is provided in this branch.
- [Wiemann and Katzfuss (2023)](https://link.springer.com/article/10.1007/s13253-023-00580-z): See branch
  [mvtm](https://github.com/katzfuss-group/batram/tree/mvtm). Wiemann and Katzfuss (2023) present a scalable Bayesian nonparametric framework that models large, multivariate, non-Gaussian spatial fields using triangular transport maps with Gaussian process components
- [Chakraborty and Katzfuss (2024)](https://arxiv.org/pdf/2409.19208): See branch [ShrinkTM](https://github.com/katzfuss-group/batram/tree/ShrinkTM). This work introduces parametric shrinkage toward a parametric base model, enabling the use of the transport map methodology in limited data scenarios, including cases with very few -- even one -- observed spatial fields.

## Installation

Run the following command in your local virtual environment. Please make sure
that a c++ compiler and the python headers are installed since this is needed to
install the dependecy [veccs](https://github.com/katzfuss-group/veccs).

1. `pip install -e .`

## Getting Started

For a quick introduction to the package, see the [getting-started notebook](notebooks/getting-started.ipynb) in the `notebooks` folder. It covers
key topics such as data preprocessing, model fitting, model evaluation, and
posterior sampling.



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
