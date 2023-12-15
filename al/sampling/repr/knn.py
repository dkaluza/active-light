import math

import faiss
import torch
from torch.utils.data import DataLoader

from al.sampling.base import ActiveState, InformativenessProto


class KNearestNeighborRepr(InformativenessProto):
    def __init__(self, k=25, batch_size=32) -> None:
        super().__init__()
        self.k = k
        self.batch_size = batch_size

    def __call__(self, state: ActiveState) -> torch.FloatTensor:
        index = self.get_index(state)
        distances = self.get_neighbor_distances(index, state)
        distances = distances.nan_to_num(1e6)
        return 1 / (1 + distances.mean(dim=1))

    def get_index(self, state: ActiveState) -> faiss.Index:
        pool = []
        for batch in self._iterate_over_pool(state=state):
            assert isinstance(batch, list)
            # assumption batch is always a tuple with features at position 0
            pool.append(batch[0])

        pool = torch.concat(pool, dim=0)

        n_features = pool.shape[1]
        n_samples = pool.shape[0]
        nlist = math.ceil(math.sqrt(n_samples))

        quantizer = faiss.IndexFlatL2(n_features)
        index = faiss.IndexIVFFlat(quantizer, n_features, nlist)
        index.nprobe = 3
        index.train(pool)
        index.add(pool)

        return index

    def get_neighbor_distances(
        self, index: faiss.Index, state: ActiveState
    ) -> torch.FloatTensor:
        distances = []

        for batch in self._iterate_over_pool(state=state):
            assert isinstance(batch, list)
            # assumption batch is always a tuple with features at position 0
            features = batch[0]

            # we are using k+1 as the point itself should be usually in index
            distances_squared, _idx = index.search(features, k=self.k + 1)
            # we are removing point itself as it will have always distance 0
            distances_squared = torch.from_numpy(distances_squared[:, 1:])
            distances.append(distances_squared**0.5)

        return torch.concat(distances, dim=0)

    def _iterate_over_pool(self, state: ActiveState):
        pool = state.get_pool()
        loader = DataLoader(pool, batch_size=self.batch_size, shuffle=False)
        yield from loader
