import torch
import networkx
from graphrl import graphrl


class ToyGraphDistribution:
    def sample_graph(self):
        features = torch.tensor([0., 0, 0, 0])
        adjacency = torch.tensor(
            [[0., 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 1], [0, 1, 1, 0]])
        weights = torch.tensor([[1.] * 4] * 4) * adjacency

        g = graphrl.Graph(features, adjacency, weights)

        return g


class ErdosRenyiDistribution:
    def sample_graph(self, size):
        features = torch.zeros(size)

        nxg = networkx.erdos_renyi_graph(size, 0.15)
        adjacency = torch.tensor(networkx.to_numpy_array(nxg))

        weights = adjacency
        g = graphrl.Graph(features, adjacency, weights)
        return g


class BarabasiAlbertDistribution:
    def sample_graph(self, size):
        features = torch.zeros(size)

        nxg = networkx.barabasi_albert_graph(size, 4)
        adjacency = torch.tensor(networkx.to_numpy_array(nxg))

        weights = adjacency
        g = graphrl.Graph(features, adjacency, weights)
        return g
