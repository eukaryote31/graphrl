from graphrl import graphgen


def test_toygen():
    gen = graphgen.ToyGraphDistribution()
    g = gen.sample_graph()

    assert len(g) == 4


def test_erdosrenyigen():
    gen = graphgen.ErdosRenyiDistribution()
    g = gen.sample_graph(50)

    assert len(g) == 50


def test_barabasialbertgen():
    gen = graphgen.BarabasiAlbertDistribution()
    g = gen.sample_graph(50)

    assert len(g) == 50
