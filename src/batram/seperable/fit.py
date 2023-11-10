import gpytorch
import torch
from tqdm import tqdm

from . import gpcov


def fit_sperable(
    data: gpcov.Data,
    subsample: None | int = None,
    niters=2000,
    lr=0.1,
    silent: bool = False,
) -> tuple[gpcov.SperableGP, gpytorch.likelihoods.GaussianLikelihood, list[float]]:
    if subsample:
        import copy

        data = copy.deepcopy(data)
        idx = torch.randperm(data.loc_sp.shape[0])[:subsample]
        data.loc_sp = data.loc_sp[idx, :]
        data.resp = data.resp[:, :, idx]
        data.eta = None
        data._resp_long = None
        data._loc_sp_long = None

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = gpcov.SperableGP(data, likelihood)
    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr
    )  # Includes GaussianLikelihood parameters
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, niters)

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    losses = []
    stopper = gpcov.EarlyStopper2(1e-5, 0)

    for i in (tqdm_iter := tqdm(range(niters), disable=silent)):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model()

        # Calc loss and backprop gradients
        loss = torch.zeros(())
        for j in range(data.nrep):
            loss += -mll(output, data.to_long()[0][j, ...])
        loss /= data.nrep

        losses.append(loss.item())
        loss.backward()
        if torch.isnan(loss):
            raise RuntimeError("loss is NaN")

        # consider early stopping
        if stopper.stop_test(loss.item()):
            print(f"stopping early at iteration {i}")
            break

        tqdm_iter.set_description(f"loss: {loss.item():.3f}")

        optimizer.step()
        scheduler.step()

    return model, likelihood, losses
