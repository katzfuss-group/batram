import torch

import batram.process_position as pp


def test_process_position_from_to():
    locs = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [-1.0, 1.0, 0.0],
            [-2.0, 2.0, 0.0],
        ]
    )
    params = pp.ProcessPosition(locs)
    locs2 = params()

    assert torch.allclose(locs, locs2, atol=1e-6)
