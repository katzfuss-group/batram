from dataclasses import dataclass

import numpy as np
import torch
import veccs.orderings
from veccs.utils import inverse_permutation


@dataclass
class VanillaData:
    """
    Vanilla data class.

    Holds $k$ replicates of $m$-variate spatial field observed at $n$ locations.

    It is assumed the data is ordered according to lexsort ordering on the locations,
    i.e., the first coordinate is sorted first, then the second, and so on.

    Attributes
    ----------
    locs
        Locations of the data. Shape (n, d)
    response
        Response of the data. Shape (k, m, n)
    """

    response: np.ndarray  # repl, proc, sp --- k, m, n
    locs: np.ndarray  # n, d

    @property
    def nrep(self) -> int:
        return self.response.shape[0]

    @property
    def nproc(self) -> int:
        return self.response.shape[1]

    @property
    def nlocs(self) -> int:
        return self.response.shape[2]

    @property
    def spdim(self) -> int:
        return self.locs.shape[1]

    def __getitem__(self, key):
        return VanillaData(self.response[key], self.locs)

    @staticmethod
    def from_unordered(response: np.ndarray, locs: np.ndarray) -> "VanillaData":
        assert response.ndim == 3
        assert locs.ndim == 2
        assert locs.shape[0] == response.shape[2]

        loc_dim = locs.shape[1]

        # sort data according to coordinates
        ord = np.lexsort([locs[:, i] for i in reversed(range(loc_dim))])
        locs = locs[ord, :]
        response = response[..., ord]

        return VanillaData(response, locs)


@dataclass
class Data:
    """
    Data class

    Holds $n$ replicates of a uni-variate spatial field observed at $N$
    locations. The data in this class has not been normalized.

    Note
    ----
    scales refers scaled distance to the nearest neighbor.

    Attributes
    ----------
    locs
        Locations of the data. shape (N, d)
    response
        Response of the data. Shape (n, N)

    augmented_response
        Augmented response of the data. Shape (n, N, m + 1). nan indicates no
        conditioning.

    conditioning_sets
        Conditioning sets of the data. Shape (N, m)

    order
        Order used to sort the data. Shape (N)
    """

    locs: torch.Tensor
    response: torch.Tensor
    augmented_response: torch.Tensor
    conditioning_sets: torch.Tensor
    order: np.ndarray

    @property
    def nlocs(self):
        return self.locs.shape[0]

    @property
    def nreps(self):
        return self.response.shape[0]

    @property
    def ncond_sets(self):
        return self.conditioning_sets.shape[1]

    @property
    def loc_dim(self):
        return self.locs.shape[1]

    @staticmethod
    def new(locs, response, conditioning_sets, order):
        """Creates a new data object.

        It is assumed that the data is ordered according to maxmin ordering.
        """
        locs = torch.as_tensor(locs, dtype=torch.float32)
        response = torch.as_tensor(response, dtype=torch.float32)
        conditioning_sets = torch.as_tensor(conditioning_sets)

        nlocs = locs.shape[0]
        ecs = torch.hstack([torch.arange(nlocs).reshape(-1, 1), conditioning_sets])
        augmented_response = torch.where(ecs == -1, torch.nan, response[:, ecs])

        return Data(locs, response, augmented_response, conditioning_sets, order)

    @staticmethod
    def new_from_unordered(
        locs: np.ndarray, response: np.ndarray, size_cond_sets: int = 30
    ) -> "Data":
        return Data.from_unordered(locs, response, size_cond_sets)

    @staticmethod
    def from_unordered(
        locs: np.ndarray, response: np.ndarray, size_cond_sets: int = 30
    ) -> "Data":
        """Creates a new data object from unordered data.

        The function uses maxmin ordering and nearest neighbors to find the
        conditioning sets. Both operations are based on L2 distance.

        Parameters:
        -----------
        locs
            Locations of the data. shape (N, d). Each row is one location in a
            d-dimensional space.

        response
            Response of the data. Shape (n, N)

        size_cond_sets
            Size of the conditioning sets.

        """

        # order data according to maxmin ordering using L2
        order = veccs.orderings.maxmin_cpp(locs)
        locs = locs[order, ...]
        response = response[..., order]

        # find nearest neighbors according to L2
        cs = veccs.orderings.find_nns_l2(locs, size_cond_sets)

        return Data.new(locs, response, cs, order)

    @staticmethod
    def from_vanilla_data(data: VanillaData, size_cond_sets: int = 30) -> "Data":
        """Creates a new data object.

        The function uses maxmin ordering and nearest neighbors to find the
        conditioning sets. Both operations are based on L2 distance.

        Parameters
        ----------
        data
            Plain data object

        size_cond_sets
            Size of the conditioning sets.
        """
        if data.nproc != 1:
            raise ValueError("Only one process is supported.")

        return Data.new_from_unordered(
            data.locs, data.response[:, 0, :], size_cond_sets
        )

    def to_vanilla(self) -> VanillaData:
        """Converts data to vanilla data."""
        locs = self.locs.numpy()
        response = np.expand_dims(self.response.numpy(), 1)

        return VanillaData.from_unordered(response, locs)


@dataclass
class MultVarData:
    """
    Multivariate data class.

    Holds $k$ replicates of $m$-variate spatial field observed at $n$ locations.
    The code assumes that $n$ is larger than 5.


    Attributes
    ----------

    response
        Response of the data. Shape (k, m x n)

    locs_sp
        Locations of the data. Shape (n, d_s)

    locs_proc
        Process positions used to sort the data. Shape (m, d_p)

    response_augmented
        Augmented response of the data. Shape (k, m x n, m + 1). nan indicates
        no conditioning.

    process_ids
        Process ids of the data. Shape (k, m x n)

    location_ids
        Location ids of the data. Shape (k, m x n)

    conditioning_sets
        Conditioning sets of the data. Shape (m x n, cs_size)

    ordering
        Ordering used to sort the data. Shape (m x n)
    """

    response: torch.Tensor
    locs_sp: torch.Tensor
    locs_proc: torch.Tensor
    response_augmented: torch.Tensor
    process_ids: torch.Tensor
    location_ids: torch.Tensor
    conditioning_sets: torch.Tensor
    ordering: np.ndarray
    order_last_var_last: bool

    @property
    def nlocs(self) -> int:
        return self.locs_sp.shape[0]

    @property
    def nreps(self) -> int:
        return self.response.shape[0]

    @property
    def nprocs(self) -> int:
        pid = self.process_ids.max().item()
        if isinstance(pid, float):
            raise RuntimeError("Process ids are not integers. {self.process_ids}.")
        return pid + 1

    @property
    def ncond_sets(self) -> int:
        return self.conditioning_sets.shape[1]

    @property
    def ordering_inverse(self) -> np.ndarray:
        return inverse_permutation(self.ordering)

    @staticmethod
    def new(
        response: np.ndarray,
        locs_sp: np.ndarray,
        locs_proc: np.ndarray,
        size_cond_sets: int,
        order_last_var_last: bool = False,
    ) -> "MultVarData":
        nrep, nproc, nlocs = response.shape
        assert len(locs_sp.shape) == 2
        assert locs_sp.shape[0] == nlocs

        if nlocs <= 5:
            raise RuntimeError(
                "The code assumes that at least 6 locations have been observed."
            )

        if size_cond_sets > nlocs - 1:
            raise RuntimeError(
                "Number of max_neighbours larger than number of actual neighbours."
            )

        # create refereces to spatial locations and processes
        process_ids = np.repeat(np.arange(nproc), nlocs)
        location_ids = np.tile(np.arange(nlocs), nproc)

        locs_comb = np.stack(
            [
                np.concatenate(
                    [
                        locs_proc[process_ids[i], :],
                        locs_sp[location_ids[i], :],
                    ]
                )
                for i in range(nproc * nlocs)
            ],
            axis=0,
        )

        # create ordering
        if not order_last_var_last:
            ord = veccs.orderings.maxmin_cpp(locs_comb)
        else:
            mask_first = process_ids != nproc - 1
            mask_last = process_ids == nproc - 1
            locs_first = locs_comb[mask_first, :]
            locs_last = locs_comb[mask_last, :]

            assert np.all(
                np.concatenate([locs_first, locs_last], 0) == locs_comb
            ), "returned order is not correct wrt locs_comb"
            ord = veccs.orderings.maxmin_pred_cpp(locs_first, locs_last)

        response_ord = response.reshape(nrep, -1)[..., ord]
        locs_comb_ord = locs_comb[ord, :]
        process_ids_ord = process_ids[ord]
        location_ids_ord = location_ids[ord]

        # find nearest neighbours according to L2
        conditioning_sets = veccs.orderings.find_nns_l2(
            locs_comb_ord, max_nn=size_cond_sets
        )

        # get augmented responses
        data_simple = Data.new(locs_comb_ord, response_ord, conditioning_sets, ord)

        return MultVarData(
            response=torch.as_tensor(response_ord, dtype=torch.float32),
            locs_sp=torch.as_tensor(locs_sp, dtype=torch.float32),
            locs_proc=torch.as_tensor(locs_proc, dtype=torch.float32),
            process_ids=torch.as_tensor(process_ids_ord),
            location_ids=torch.as_tensor(location_ids_ord),
            conditioning_sets=conditioning_sets,
            response_augmented=data_simple.augmented_response,
            ordering=ord,
            order_last_var_last=order_last_var_last,
        )

    @staticmethod
    def from_vanilla(
        data: VanillaData,
        locs_proc: np.ndarray,
        size_cond_sets: int = 30,
        order_last_var_last: bool = False,
    ) -> "MultVarData":
        """
        Creates a new data object.

        Creates a new data object from vanilla data together with given porcess
        locations. Sorts the data according to maxmin ordering and finds the
        conditioning sets using L2 distance.

        Observations are assumed to be embedded in a pseudo space where each
        location is augmented by the process locations.
        """

        mvdat = MultVarData.new(
            response=data.response,
            locs_sp=data.locs,
            locs_proc=locs_proc,
            size_cond_sets=size_cond_sets,
            order_last_var_last=order_last_var_last,
        )

        return mvdat

    def to_vanilla(self) -> VanillaData:
        """Converts data to vanilla data."""

        responses: list[np.ndarray] = []
        locs_list: list[np.ndarray] = []
        for i in range(self.nprocs):
            mask = self.process_ids == i
            locs = self.locs_sp[self.location_ids[mask], :]
            response = self.response[:, mask]
            ord = np.lexsort([locs[:, i] for i in reversed(range(locs.shape[1]))])

            locs_ord = locs[ord, :].numpy()
            response_ord = response[:, ord].numpy()

            responses.append(response_ord)
            locs_list.append(locs_ord)

        if len(locs) > 1:
            for i in range(1, len(locs_list)):
                if not np.array_equal(locs_list[i], locs_list[0]):
                    raise ValueError("Process locations are not equal.")

        stacked_response = np.stack(responses, axis=1)
        return VanillaData(stacked_response, locs_list[0])

    def data_per_process(self) -> list[Data]:
        """Returns a list of Data objects, one for each process."""

        datas = []
        for i in range(self.nprocs):
            mask = self.process_ids == i
            locs = self.locs_sp[self.location_ids[mask], :]
            responses = self.response[:, mask]
            datas.append(
                Data.new_from_unordered(
                    locs.numpy(), responses.numpy(), self.ncond_sets
                )
            )
        return datas

    def reorder(self, new_process_pos: torch.Tensor) -> "MultVarData":
        size_cond_sets = self.ncond_sets
        vd = self.to_vanilla()
        return MultVarData.from_vanilla(
            vd, new_process_pos.numpy(), size_cond_sets, self.order_last_var_last
        )
