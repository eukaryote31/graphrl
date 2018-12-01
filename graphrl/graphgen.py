import torch
import networkx
import random
from graphrl import graphrl


class Distribution:
    def __init__(self, device='cpu'):
        self.dev = device

    def mk_weights(self, size, adjacency, mode):
        if mode == 'ones':
            return adjacency
        elif mode == 'uniform':
            return torch.tensor((size, size), device=self.dev).uniform_(0, 1)

    def sample_graph(self):
        raise NotImplementedError


class ToyGraphDistribution(Distribution):
    def __init__(self, device='cpu'):
        super().__init__(device)

    def sample_graph(self):
        adjacency = torch.tensor(
            [
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [1, 0, 0, 1, 1, 0, 0, 0],
                [0, 1, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 1, 1, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 1, 0],
            ], dtype=torch.float32, device=self.dev)
        weights = torch.tensor([[1.] * 8] * 8, device=self.dev) * adjacency

        g = graphrl.Graph(adjacency, weights)

        return g


class ErdosRenyiDistribution(Distribution):
    def __init__(self, device='cpu'):
        super().__init__(device)

    def sample_graph(self, size=50, mode='ones'):
        nxg = networkx.erdos_renyi_graph(size, 0.15)
        adjacency = torch.tensor(networkx.to_numpy_array(
            nxg), dtype=torch.get_default_dtype(), device=self.dev)

        weights = self.mk_weights(size, adjacency, mode)
        g = graphrl.Graph(adjacency, weights)
        return g


class BarabasiAlbertDistribution(Distribution):
    def __init__(self, device='cpu'):
        super().__init__(device)

    def sample_graph(self, size=50, mode='ones'):
        nxg = networkx.barabasi_albert_graph(size, 4)
        adjacency = torch.tensor(networkx.to_numpy_array(
            nxg), dtype=torch.get_default_dtype(), device=self.dev)

        weights = self.mk_weights(size, adjacency, mode)

        g = graphrl.Graph(adjacency, weights)
        return g


class MixedDistribution(Distribution):
    def __init__(self, mode='ones', sizerange=(50, 100), device='cpu'):
        super().__init__(device)
        self.mode = mode
        self.distributions = [
            ErdosRenyiDistribution(device),
            BarabasiAlbertDistribution(device)
        ]
        self.sizerange = sizerange

    def sample_graph(self):
        distr = random.choice(self.distributions)
        return distr.sample_graph(size=random.randint(*self.sizerange), mode=self.mode)
