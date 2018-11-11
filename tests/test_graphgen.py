from graphrl import graphgen
import torch


def test_toygen():
    gen = graphgen.ToyGraphDistribution()
    g = gen.sample_graph()

    assert torch.all(g.adjacency.transpose(0, 1) == g.adjacency)
    assert len(g) == 4


def test_erdosrenyigen():
    gen = graphgen.ErdosRenyiDistribution()
    g = gen.sample_graph(50)

    assert torch.all(g.adjacency.transpose(0, 1) == g.adjacency)
    assert len(g) == 50


def test_barabasialbertgen():
    gen = graphgen.BarabasiAlbertDistribution()
    g = gen.sample_graph(50)

    assert torch.all(g.adjacency.transpose(0, 1) == g.adjacency)
    assert len(g) == 50
