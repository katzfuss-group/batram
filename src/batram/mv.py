import torch

from .data import MultVarData
from .legmods import Data, SimpleTM
from .process_position import ProcessPosition


class DataModule(torch.nn.Module):
    def __init__(self, process_positions: torch.Tensor) -> None:
        super().__init__()

        self.process_position = ProcessPosition(process_positions)

    def forward(self, data: MultVarData) -> Data:
        proc_pos = self.process_position()
        nprocs = self.process_position.num_procs
        nlocs = data.nlocs
        nauglocs = nprocs * nlocs

        locs_augmented = torch.stack(
            [
                torch.concatenate(
                    [
                        proc_pos[data.process_ids[i], :],
                        data.locs_sp[data.location_ids[i], :],
                    ]
                )
                for i in range(nauglocs)
            ],
            dim=0,
        )

        return Data(
            locs=locs_augmented,
            response=data.response,
            conditioning_sets=data.conditioning_sets,
            augmented_response=data.response_augmented,
            order=data.ordering,
        )


class MultVarTM(SimpleTM):
    def __init__(
        self,
        data: MultVarData,
        process_pos_init: torch.Tensor,
        theta_init: torch.Tensor,
        linear=False,
        smooth: float = 1.5,
        nugMult: float = 4,
        new_method: bool = True,
    ) -> None:
        data_mod = DataModule(process_pos_init)
        data_pseudo = data_mod(data)
        super().__init__(data_pseudo, theta_init, linear, smooth, nugMult, new_method)

        self.data_mv = data
        self.data_mod = data_mod

    @property
    def data(self) -> Data:
        return self.data_mod(self.data_mv)

    def forward(
        self, batch_idx: None | torch.Tensor = None, data: None | MultVarData = None
    ) -> torch.Tensor:
        if data is None:
            data = self.data_mv

        data_pseudo_space = self.data_mod(data)
        intloglik = super().forward(batch_idx, data_pseudo_space)
        return intloglik
