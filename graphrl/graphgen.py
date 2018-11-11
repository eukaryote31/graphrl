import torch
import networkx
from graphrl import graphrl


class Distribution:
    def mk_weights(self, size, adjacency, mode):
        if mode == 'ones':
            return adjacency
        elif mode == 'uniform':
            return torch.tensor((size, size)).uniform_(0, 1)


class ToyGraphDistribution(Distribution):
    def sample_graph(self):
        adjacency = torch.tensor(
            [[0., 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 1], [0, 1, 1, 0]])
        weights = torch.tensor([[1.] * 4] * 4) * adjacency

        g = graphrl.Graph(adjacency, weights)

        return g


class ErdosRenyiDistribution(Distribution):
    def sample_graph(self, size, mode='ones'):
        nxg = networkx.erdos_renyi_graph(size, 0.15)
        adjacency = torch.tensor(networkx.to_numpy_array(nxg), dtype=torch.get_default_dtype())

        weights = self.mk_weights(size, adjacency, mode)
        g = graphrl.Graph(adjacency, weights)
        return g


class BarabasiAlbertDistribution(Distribution):
    def sample_graph(self, size, mode='ones'):
        nxg = networkx.barabasi_albert_graph(size, 4)
        adjacency = torch.tensor(networkx.to_numpy_array(nxg))

        weights = self.mk_weights(size, adjacency, mode)
        g = graphrl.Graph(adjacency, weights)
        return g
