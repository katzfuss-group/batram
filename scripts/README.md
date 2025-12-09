# Paper reproduction

To reproduce figures and results of the paper, follow the instructions here. First, to verify your installation and to get used to the package usage, try running all the cells in the [getting started notebook](../notebooks/tutorial-mf.ipynb).

Next, we download the GCM-RCM dataset we use in the paper. We will use the script `get_gcm_rcm.py`, which needs some additional dependencies, which can be installed via conda by running. It is also included in the [data folder](../tests/data) for convenience. 

```
conda install -c conda-forge xarray dask netCDF4 bottleneck
conda update -c conda-forge ca-certificates certifi openssl requests urllib3
```

Two important things to note from the script before you run:
1) SSL certificate for the GCM data was expired at the time of writing this note (11/14/2025), so we ignore those warnings to get the data. However, the script is written in such a way that we try to safe download (with certificates) first, and if that fails we ignore the warnings.
2) The files for GCMs are somewhat big (around $900$ MBs for each ensemble member). They contain a lot of time and space points we are not interested in this analysis, so we download them locally, subset them, and erase the downloaded file. You end up with four moderately sized files (two locations, two observations), not surpassing $40$ MBs.

Now, running the downloader for the specific day we prespecify (01-25-1996, this could be changed as needed in the script), run `python get_gcm_rcm.py`. The files will be downloaded this [data folder](../tests/data), since some scripts/notebooks assume they are there. We download max daily temperature, but you can also specify another variable included in both GCM and RCM models.

Finally, the figures themselves are reproduced in different notebooks. You need to install `seaborn` for some visualizations. 
- Figures 1, 8 (example ensemble members form GCM-RCM pairs, samples from the model) in the [climate notebook](../notebooks/climate-example.ipynb)
- Figure 2 (illustration  of the conditional maximin order) in the [maximin order notebook](../notebooks/maxmin-order-plot.ipynb)
- Figure 3 (intuition that will come handy for model parametrization) in the [intuition notebook](../notebooks/intuition.ipynb)
- Figure 4 (model performance/sample when there is a linear relationship, i.e., block averages, between fidelities) in the [linear notebook](../notebooks/linear.ipynb)
- Figures 5, 9 (log-scores): To get the actual logscores computed from the different considered models, you have to run the scripts that are named `run_ensemble_{method}_{experiment}.py`, replacing the appropriate method and experiment (methods are `mf, matern, hk, nargp, VAE` and experiments are `linear, min, climate`). This will create some `.csv` files in the `./results` directory. Finally, in the [plot notebook](../notebooks/plot-logscores.ipynb) you can reproduce the Figures. Files with log-scores included in the repo, but you should be able to reproduce them. 
- Figure 6 (model performance when there is a non-linear, i.e., block minima between fidelities) in the [nonlinear notebook](../notebooks/nonlinear.ipynb). Very similar than the linear notebook but with less diagnostics about the model. 
