import math

import faiss
import faiss.contrib.torch_utils  # import needed to add faiss torch interoperability
import torch
from torch.utils.data import DataLoader

from al.base import get_default_torch_device
from al.sampling.base import ActiveState, InformativenessProto


class KNearestNeighborRepr(InformativenessProto):
    CACHE_KEY = "KNN_pool_index"

    def __init__(self, k=25, batch_size=32) -> None:
        super().__init__()
        self.k = k
        self.batch_size = batch_size
        self.index_device = None

    def __call__(self, state: ActiveState) -> torch.FloatTensor:
        if self.index_device is None:
            self.index_device = get_default_torch_device()

        pool = []
        for batch in self._iterate_over_pool(state=state):
            assert isinstance(batch, list)
            # assumption batch is always a tuple with features at position 0
            pool.append(batch[0])

        pool = torch.concat(pool, dim=0).to(self.index_device)

        index = self._get_index_from_cache(state)
        if index is None:
            index = self.build_index(pool)
            self._save_index_in_cache(state, index)

        distances = self.get_neighbor_distances(index, pool)
        distances = distances.nan_to_num(1e6)  # note: reconsider
        return 1 / (1 + distances.mean(dim=1))

    def _get_index_from_cache(self, state: ActiveState) -> faiss.Index | None:
        return state.get_from_cache(self.CACHE_KEY)

    def _save_index_in_cache(self, state: ActiveState, index: faiss.Index):
        state.save_in_cache(self.CACHE_KEY, index)

    def build_index(self, pool: torch.Tensor) -> faiss.Index:
        n_features = pool.shape[1]
        n_samples = pool.shape[0]
        nlist = math.sqrt(n_samples) if n_samples > 5_000 else math.log2(n_samples)
        nlist = math.ceil(nlist)

        index = (
            self._get_gpu_index(n_features=n_features, nlist=nlist)
            if self.index_device.type == "cuda"
            else self._get_cpu_index(n_features=n_features, nlist=nlist)
        )

        index.nprobe = 3
        pool = pool.float()
        index.train(pool)
        index.add(pool)
        return index

    def _get_cpu_index(self, n_features, nlist):
        quantizer = faiss.IndexFlatL2(n_features)
        index = faiss.IndexIVFFlat(quantizer, n_features, nlist)
        return index

    def _get_gpu_index(self, n_features, nlist):
        res = faiss.StandardGpuResources()
        index = self._get_cpu_index(n_features=n_features, nlist=nlist)
        index = faiss.index_cpu_to_gpu(res, device=self.index_device.index, index=index)

        return index

    def get_neighbor_distances(
        self, index: faiss.Index, pool: torch.Tensor
    ) -> torch.FloatTensor:
        # we are using k+1 as the point itself should be usually in index
        distances_squared, _idx = index.search(
            pool.to(self.index_device).float(), k=self.k + 1
        )
        # we are removing point itself as it will have always distance 0
        return distances_squared[:, 1:] ** 0.5

    def _iterate_over_pool(self, state: ActiveState):
        pool = state.get_pool()
        loader = DataLoader(pool, batch_size=self.batch_size, shuffle=False)
        yield from loader


k_nearest_neighbor_repr = KNearestNeighborRepr()
