import pytest
import torch

import batram.legmods as legmods


# @pytest.mark.skip("Bug")
def test_legmods_old_intlikelihood() -> None:
    kernel_result = legmods.KernelResult(
        G=torch.ones(2, 2).diag_embed(),
        GChol=torch.ones(2, 2).diag_embed(),
        nug_mean=torch.ones(2, 1, 1),
    )
    data = legmods.Data.new(
        locs=torch.ones(2, 2),
        response=torch.ones(2, 2, 1),
        conditioning_set=-torch.ones(2, 1).long(),
    )
    augdata = legmods.AugmentedData(
        data_size=2,
        batch_size=2,
        batch_idx=torch.arange(2),
        locs=torch.ones(2, 2),
        augmented_response=torch.ones(2, 2, 2),
        scales=torch.ones(2, 1, 1),
        data=data,
    )

    intlike = legmods.IntLogLik(1.0)
    with torch.no_grad():
        intlik = intlike.forward(augdata, kernel_result)
    assert intlik.sum().item() == pytest.approx(-2.43279)
