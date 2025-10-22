from typing import Any

import numpy as np
import torch
from scipy.optimize import minimize
from tqdm import tqdm

from .legmods import Data, FitResult, SimpleTM
from .stopper import PEarlyStopper

Array = Any


# --- helper utilities ---------------------------------------------------------


def _pack_parameters(module: torch.nn.Module):
    """
    Flatten all parameters of `module` to a single 1D NumPy array and
    return (flat, meta) where meta holds reshape info for each param.
    """
    flats = []
    meta = []  # (name, shape, dtype, device, numel)
    with torch.no_grad():
        for name, p in module.named_parameters():
            arr = p.detach().cpu().numpy().ravel()
            flats.append(arr)
            meta.append((name, tuple(p.shape), p.dtype, p.device, arr.size))
    x0 = np.concatenate(flats) if flats else np.array([], dtype=np.float64)
    return x0, meta


def _unpack_and_load(module: torch.nn.Module, x: np.ndarray, meta):
    """
    Load a flat NumPy vector `x` back into module parameters using `meta`.
    """
    with torch.no_grad():
        offset = 0
        for (name, shape, dtype, device, numel), (_, p) in zip(
            meta, module.named_parameters()
        ):
            chunk = x[offset : offset + numel].reshape(shape)
            tensor = torch.from_numpy(chunk).to(device=device, dtype=dtype)
            p.copy_(tensor)
            offset += numel


def _clone_state_dict(module: torch.nn.Module):
    # create a detached clone of the entire state_dict (including buffers)
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


# --- Nelder–Mead fit ----------------------------------------------------------


def fit_nelder_mead(
    model: SimpleTM,
    num_iter: int,
    test_data: Data | None = None,
    stopper: PEarlyStopper | None = None,
    silent: bool = False,
    initial_simplex: np.ndarray | None = None,  # optional: pass a custom simplex
    ftol: float | None = None,  # optional: override SciPy defaults
    xtol: float | None = None,  # optional: override SciPy defaults
):
    """
    Fit the model with SciPy's Nelder–Mead (gradient-free).

    Parameters
    ----------
    num_iter
        Maximum number of Nelder–Mead iterations (maps to SciPy `maxiter`).
    batch_size
        Ignored – NM evaluates full loss (kept for interface compatibility).
    test_data
        Data to use for testing. If None, do not test.
    stopper
        Early stopper evaluated on `test_data` after each iteration (requires test data).
        If triggered, we restore the best state and stop.
    silent
        If True, do not print progress.
    initial_simplex
        Optional custom initial simplex (SciPy `initial_simplex`).
    ftol, xtol
        Optional tolerances to pass to SciPy (float).
    """
    self = model

    if stopper is not None and test_data is None:
        raise ValueError("Cannot use stopper without test data.")

    # Nelder–Mead will overwrite parameters frequently; pre-pack them once.
    x0, meta = _pack_parameters(self)

    # losses & chains (track once per iteration, not per function eval)
    # evaluate initial loss before optimization
    with torch.no_grad():
        init_loss = float(self().item())
    losses: list[float] = [init_loss]
    test_losses: list[float] = (
        [] if test_data is None else [float(self(data=test_data).item())]
    )

    # store initial parameter/value snapshots
    parameters = [
        {k: np.copy(v.detach().cpu().numpy()) for k, v in self.named_parameters()}
    ]
    values = [
        {k: np.copy(v.detach().cpu().numpy()) for k, v in self.named_tracked_values()}
    ]

    # tqdm progress bar per "iteration" (SciPy-defined, not function eval)
    pbar = tqdm(total=num_iter, disable=silent)

    # keep best state for early stopping (if any)
    best_state = _clone_state_dict(self)
    best_test = test_losses[-1] if test_losses else np.inf

    # Objective: set weights from x and compute full-data loss
    def objective(x: np.ndarray) -> float:
        _unpack_and_load(self, x, meta)
        with torch.no_grad():
            return float(self().item())

    # Callback runs once per Nelder–Mead iteration; we log + early stop here
    def callback(xk: np.ndarray, *_):
        nonlocal best_state, best_test

        # record current state (already loaded by SciPy before callback)
        with torch.no_grad():
            cur_loss = float(self().item())
        losses.append(cur_loss)

        # test/validation
        desc = f"Train Loss: {losses[-1]:.3f}"
        if test_data is not None:
            with torch.no_grad():
                cur_test = float(self(data=test_data).item())
            test_losses.append(cur_test)
            desc += f", Test Loss: {cur_test:.3f}"

            # Early stopping
            if stopper is not None:
                state = _clone_state_dict(self)
                stop = stopper.step(cur_test, state)
                if cur_test < best_test:
                    best_test = cur_test
                    best_state = _clone_state_dict(self)
                if stop:
                    # Returning True from callback asks SciPy to terminate early
                    pbar.set_description(desc + " (early stop)")
                    pbar.close()
                    return True  # type: ignore[return-value]

        # store parameter & tracked values snapshot for this iteration
        parameters.append(
            {k: np.copy(v.detach().cpu().numpy()) for k, v in self.named_parameters()}
        )
        values.append(
            {
                k: np.copy(v.detach().cpu().numpy())
                for k, v in self.named_tracked_values()
            }
        )

        pbar.update(1)
        pbar.set_description(desc)

        # Respect user-specified max iterations in case SciPy overshoots;
        # ask SciPy to stop if we've logged `num_iter` iterations.
        if len(losses) - 1 >= num_iter:  # minus 1 because we seeded with initial loss
            pbar.set_description(desc + " (max iter)")
            pbar.close()
            return True  # type: ignore[return-value]

        return None

    # Build SciPy options
    options = {
        "maxiter": int(num_iter),
        "initial_simplex": initial_simplex,
    }
    # prune None values; SciPy doesn't like them present
    options = {k: v for k, v in options.items() if v is not None}

    # Extra tol kwargs (SciPy expects them at top-level, not in options)
    minimize_kwargs = {}
    if ftol is not None:
        minimize_kwargs["tol"] = (
            ftol  # SciPy's Nelder–Mead mainly respects `xatol`/`fatol` in options in newer versions; `tol` still commonly used
        )
    # x/func tolerances via options if you prefer explicit control:
    # options["xatol"] = xtol if xtol is not None else options.get("xatol", None)
    if xtol is not None:
        options["xatol"] = xtol

    # Run Nelder–Mead
    res = minimize(
        fun=objective,
        x0=x0,
        method="Nelder-Mead",
        callback=callback,
        options=options,
        **minimize_kwargs,
    )

    # Close tqdm if SciPy terminated without hitting our callback again
    if hasattr(pbar, "disable") and not pbar.disable:
        pbar.close()

    # If early-stopper requested the best state, restore it;
    # otherwise load the optimizer's final solution.
    if stopper is not None and test_data is not None:
        self.load_state_dict(best_state)
    else:
        _unpack_and_load(self, res.x, meta)

    # Build chains for return (stack recorded snapshots)
    param_chain = {}
    for k in parameters[0].keys():
        param_chain[k] = np.stack([d[k] for d in parameters], axis=0)

    tracked_chain = {}
    for k in values[0].keys():
        tracked_chain[k] = np.stack([d[k] for d in values], axis=0)

    return FitResult(
        model=self,
        max_m=self.data.conditioning_sets.shape[-1],
        losses=np.array(losses),
        parameters=parameters[-1],
        test_losses=np.array(test_losses) if test_data is not None else None,
        param_chain=param_chain,
        tracked_chain=tracked_chain,
    )
