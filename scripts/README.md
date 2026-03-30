# Paper reproduction

To reproduce the figures and results of the paper, follow the instructions below. As a first check that your installation is working, run all cells in the [getting started notebook](../notebooks/tutorial-mf.ipynb).

## Download the climate data

We use the GCM-RCM dataset in the paper. The downloader script is `get_gcm_rcm.py`, which requires a few additional dependencies:

```bash
conda install -c conda-forge xarray dask netCDF4 bottleneck
conda update -c conda-forge ca-certificates certifi openssl requests urllib3
```

A few practical notes before running the downloader:

1. At the time this README was originally written, the SSL certificate for the GCM source had expired. The download script therefore first attempts a standard secure download, and only falls back to ignoring certificate warnings if that fails.
2. The raw GCM files are fairly large (around 900 MB per ensemble member), but the script subsets them to the portion needed for the paper and removes the large temporary files afterwards. The resulting files stored locally are much smaller.

To download the prespecified day used in the paper (01-25-1996), run:

```bash
python get_gcm_rcm.py
```

The processed files are saved in the [data folder](../tests/data), since several scripts and notebooks assume that location. The current configuration downloads daily maximum temperature, though the script can be adapted to other variables present in both the GCM and RCM products.

## Reproduce the paper figures

Some of the paper figures are generated directly from notebooks:

- Figures 1 and 8 (example GCM-RCM ensemble members and model samples): [climate notebook](../notebooks/climate-example.ipynb)
- Figure 2 (conditional maximin ordering illustration): [maximin order notebook](../notebooks/maxmin-order-plot.ipynb)
- Figure 3 (intuition for the model parametrization): [intuition notebook](../notebooks/intuition.ipynb)
- Figure 4 (linear relationship across fidelities, using block averages): [linear notebook](../notebooks/linear.ipynb)
- Figure 5 (log-scores comparison, pre-saved log-scores, see below how to reproduce them): [logscores notebook](../notebooks/linear.ipynb)
- Figure 6 (nonlinear relationship across fidelities, using block minima): [nonlinear notebook](../notebooks/nonlinear.ipynb)
- Mini-batching usage is illustrated in the [mini-batching notebook](../notebooks/climate-example-mb.ipynb)

For some visualizations you may also need:

```bash
pip install seaborn
```

## Reproduce the log-score experiments

The experiment runners are

- `run_matern.py`
- `run_hk.py`
- `run_mf.py`
- `run_nargp.py`
- `run_vae.py`

Each script is controlled from the command line through two main options:

- `--experiment` chooses which experiment to run
- `--include-all-logscores` controls whether the output CSV stores every individual log-score or only grouped means

The supported experiments are:

- `linear`
- `min`
- `climate`

### Basic usage

Run a model on a given experiment with:

```bash
python run_mf.py --experiment linear
python run_nargp.py --experiment min
python run_hk.py --experiment climate
```

If you want the output file to contain **all individual log-scores**, include the flag:

```bash
python run_mf.py --experiment climate --include-all-logscores
```

If you omit that flag, the script writes a more compact results file containing only the corresponding means.

### MF linear variant

The `mflinear` case is handled inside `run_mf.py` rather than through a separate top-level script. Use:

```bash
python run_mf.py --experiment linear --variant mflinear
```

or, for example,

```bash
python run_mf.py --experiment climate --variant mflinear --include-all-logscores
```

### Output files

The scripts write `.csv` result files to the `results` directory. These files can then be used in the [plot notebook](../notebooks/plot-logscores.ipynb) to reproduce the log-score figures.
