# batram

This repository contains a Python implementation of the transport map method, `ShrinkTM` described in
[Chakraborty and Katzfuss (2024)](https://arxiv.org/pdf/2409.19208).

## Installation

Run the following command in your local virtual environment. Please make sure
that a c++ compiler and the python headers are installed since this is needed to
install the dependecy [veccs](https://github.com/katzfuss-group/veccs).

1. `pip install -e .`

### Remarks

- You must not install legacy [batram](https://github.com/katzfuss-group/batram/tree/main) package separately; all the codes are directly available through the installation of this package. Installation of the legacy `batram` package will create version conflict.

- Please make sure that  all the prerequisite `Python` models along with `batram` is installed and setup properly in your system. We have seen in our experiments that `Python version >= 3.11` creates computational issues in the computations, and downgrading module versions is not straightforward. Hence, we recommend sticking to `Python 3.10`.

## Getting started

- For learning how to use `ShrinkTM`, refer to  the [tutorial](notebooks/getting-started.ipynb).
- For replicating results from the preprint, first produce the simulated samples.
  - For LR900, use `python scripts/make_data.py --output LR --n_samples 300`.
  - For NR900, use `python scripts/make_data.py --output NR --n_samples 300`.
- Once you generate samples, run a [notebook](notebooks/fit-esitmable-shrinkage-tm.ipynb) to perform the experiments and the other [one](notebooks/sim_plots.ipynb) to plot the results.
- For climate data application, follow the same steps with [climate-data-application](notebooks/climate-data-application.ipynb) and [climate_plots](notebooks/climate_plots.ipynb).

## Contact

[Anirban Chakraborty](https://sites.google.com/view/anirban-chakraborty/home), for package development.

[Matthias Katzfuss](https://sites.google.com/view/katzfuss/home), to learn more about group's research.
