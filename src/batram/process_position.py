import math
import typing

import numpy as np
import torch


class ProcessPosition(torch.nn.Module):
    def __init__(self, locations_init: torch.Tensor) -> None:
        super().__init__()
        q_mat, params_unconst = self._locations_to_unconstraint(locations_init)
        self.q_mat = q_mat
        self.params = torch.nn.Parameter(params_unconst)
        self._tracked_values: dict["str", torch.Tensor] = {}

    @property
    def num_procs(self) -> int:
        return self.q_mat.shape[0] + 1

    @property
    def dim(self) -> int:
        return self.q_mat.shape[1]

    @staticmethod
    def _locations_to_unconstraint(
        locations: torch.Tensor, use_max_dim: bool = True, fix_zeros: bool = True
    ):
        # first row is 0
        assert np.allclose(locations[0, :].numpy(), 0)

        # rest
        locations = locations[1:, :]

        if use_max_dim and locations.shape[0] > locations.shape[1]:
            new_cols = locations.shape[0] - locations.shape[1]
            locations = torch.hstack(
                (locations, torch.zeros((locations.shape[0], new_cols)))
            )
            idx = np.diag_indices_from(locations)
            diag = torch.where(torch.diag(locations) == 0, 1e-6, torch.diag(locations))
            locations[idx] = diag

        locs_non_zero = locations
        q, r = torch.linalg.qr(locs_non_zero, mode="complete")
        sgn = torch.sign(torch.diag(r))
        sgn_diag = torch.diag(sgn)
        q_ = q @ sgn_diag
        r_ = sgn_diag @ r

        idx = np.diag_indices_from(r_)
        if fix_zeros:
            r_[idx] = torch.where(r_[idx] == 0.0, 1e-6, r_[idx])

        r_[idx] = torch.log(r_[idx])
        idx2 = np.triu_indices_from(r_)
        unconst_params = r_[idx2]

        return q_, unconst_params

    def _unconstraint_to_locations(self, unconst_params: torch.Tensor):
        # the first row is 0

        # the rest
        s = unconst_params.shape[0]
        n = int(-0.5 + math.sqrt(1 / 4 + 2 * s))  # inverse of n * (n + 1)/2 = s

        locs_non_zero = torch.zeros((n, n))

        # get indices, np types required, cast is no-op
        locs_non_zero_tc_np = typing.cast(np.ndarray, locs_non_zero)
        idx = np.triu_indices_from(locs_non_zero_tc_np)
        idx2 = np.diag_indices_from(locs_non_zero_tc_np)

        locs_non_zero[idx] = unconst_params
        locs_non_zero[idx2] = torch.exp(locs_non_zero[idx2])

        locs_non_zero = self.q_mat @ locs_non_zero

        locs = torch.vstack((torch.zeros((1, self.dim)), locs_non_zero))
        self._tracked_values["locations"] = locs
        return locs

    def forward(self):
        return self._unconstraint_to_locations(self.params)
