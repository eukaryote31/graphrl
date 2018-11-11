from graphrl import graphgen


def test_erdosrenyigen():
    gen = graphgen.ErdosRenyiDistribution()
    g = gen.sample_graph(50)

    assert len(g) == 50
