import torch
from graphrl import struct2vec, graphrl, graphgen


def test_s2v_sanity():
    g = graphgen.ToyGraphDistribution().sample_graph()
    s = graphrl.Solution(g)

    ge = struct2vec.GraphEmbedder()
    qvs, embs = ge([s])
    assert qvs.size() == torch.Size((1, 8))
    assert embs.size() == torch.Size((1, 8, 10))


def test_s2v_multi():
    ge = struct2vec.GraphEmbedder()
    gen = graphgen.ErdosRenyiDistribution()

    gs = []
    for _ in range(5):
        gs.append(graphrl.Solution(gen.sample_graph(12)))
    qvs, embs = ge(gs)
    assert qvs.size() == torch.Size((5, 12))
    assert embs.size() == torch.Size((5, 12, 10))
