from graphrl import graphrl, graphgen
import torch
import pytest
from unittest.mock import MagicMock


def test_mvc_terminate():
    prob = graphrl.MVCProblem()

    g = graphgen.ToyGraphDistribution().sample_graph()
    s = graphrl.Solution(g)

    s = s.add_node(0)

    assert not prob.terminate(s)

    s = s.add_node(3)
    s = s.add_node(4)

    assert not prob.terminate(s)

    s = s.add_node(6)

    assert prob.terminate(s)

    s = s.add_node(7)

    assert prob.terminate(s)

    assert prob.terminate(s.subsolution_at_step(4))
    assert not prob.terminate(s.subsolution_at_step(3))

    assert not prob.terminate(s.subsolution_at_step(1))


def test_mvc_reward():
    prob = graphrl.MVCProblem()

    g = graphgen.ToyGraphDistribution().sample_graph()
    s = graphrl.Solution(g)

    s = s.add_node(0)

    assert prob.cost(s) == -1

    s = s.add_node(3)
    s = s.add_node(2)

    assert prob.cumul_reward(s, 0, 3) == -3


def test_solution_subsolution():
    g = graphgen.ToyGraphDistribution().sample_graph()
    s = graphrl.Solution(g)

    s = s.add_node(0)
    s_at1 = s
    s = s.add_node(2)
    s_at2 = s
    s = s.add_node(3)
    s = s.add_node(1)

    s2 = graphrl.Solution(g)
    s2 = s2.add_node(0)

    assert s == s
    assert g == g
    assert s_at1 == s2
    assert s_at2 != s2
    assert s_at1 != s
    assert s_at2 != s
    assert s_at1 == s.subsolution_at_step(1)
    assert s_at2 == s.subsolution_at_step(2)
    assert s_at1 != s.subsolution_at_step(2)
    assert s.subsolution_at_step(0) == graphrl.Solution(g)
    assert s.subsolution_at_step(1) == graphrl.Solution(g).add_node(0)
    assert s.subsolution_at_step(1) != graphrl.Solution(g).add_node(2)
    assert s.subsolution_at_step(2) == graphrl.Solution(
        g).add_node(0).add_node(2)
    i = 0
    for _ in s.subsolution_at_step(1):
        i += 1
    assert i == 1


def test_view_resolution():
    g = graphgen.ToyGraphDistribution().sample_graph()
    s = graphrl.Solution(g)

    s = s.add_node(0)
    s = s.add_node(2)
    s = s.add_node(3)
    s = s.add_node(1)

    s1 = s.subsolution_at_step(1)
    assert s == s1.add_node(2).add_node(3).add_node(1)


def test_pick_random_node():
    g = graphgen.ToyGraphDistribution().sample_graph()
    s = graphrl.Solution(g)

    for _ in range(100):
        assert s.pick_random_node() < len(g)
    s = s.add_node(0)
    s = s.add_node(2)
    s = s.add_node(3)
    s = s.add_node(4)
    s = s.add_node(5)
    s = s.add_node(6)
    s = s.add_node(7)

    assert s.pick_random_node() == 1
    s = s.add_node(1)
    with pytest.raises(IndexError):
        s.pick_random_node()


def test_solution_pick_node():
    g = graphgen.ToyGraphDistribution().sample_graph()
    s = graphrl.Solution(g)

    emb = MagicMock(return_value=(torch.tensor(
        [[0, 3, 0, 0, 1, 0, 2, 0]], dtype=torch.get_default_dtype()), None))
    assert s.pick_node(emb, 0) == 1
    s = s.add_node(1)
    assert s.pick_node(emb, 0) == 6
    s = s.add_node(6)
    assert s.pick_node(emb, 0) == 4
    s = s.add_node(4)
    assert s.pick_node(emb, 0) == 0
    s = s.add_node(0)
    assert s.pick_node(emb, 0) == 2
    s = s.add_node(2)
    assert s.pick_node(emb, 0) == 3
    s = s.add_node(3)
    s = s.add_node(5)
    s = s.add_node(7)
    with pytest.raises(IndexError):
        s.pick_node(emb, 0)


def test_replaymem():
    rm = graphrl.ReplayMemory(10)

    assert len(rm) == 0

    for i in range(101):
        rm.add(i)

    assert len(rm) == 10
    print(rm)
    rm2 = graphrl.ReplayMemory(10)
    for e in [100, 91, 92, 93, 94, 95, 96, 97, 98, 99]:
        rm2.add(e)
    assert rm == rm2
