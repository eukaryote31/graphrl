from graphrl import graphrl
import torch


def test_mvc_terminate():
    prob = graphrl.MVCProblem()

    features = torch.tensor([0., 0, 0, 0])
    adjacency = torch.tensor(
        [[0., 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 1], [0, 1, 1, 0]])
    weights = torch.tensor([[1.] * 4] * 4) * adjacency

    g = graphrl.Graph(features, adjacency, weights)
    s = graphrl.State(g)

    s.add_node(0)

    assert not prob.terminate(g, s)

    s.add_node(3)

    assert prob.terminate(g, s)


def test_mvc_reward():
    prob = graphrl.MVCProblem()

    features = torch.tensor([0., 0, 0, 0])
    adjacency = torch.tensor(
        [[0., 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 1], [0, 1, 1, 0]])
    weights = torch.tensor([[1.] * 4] * 4) * adjacency

    g = graphrl.Graph(features, adjacency, weights)
    s = graphrl.State(g)

    s.add_node(0)

    assert prob.reward(g, s, 3) == -1

    s.add_node(3)
    s.add_node(2)

    assert prob.cumul_reward(g, s, 0, 3) == -3


def test_state_substate():
    features = torch.tensor([0., 0, 0, 0])
    adjacency = torch.tensor(
        [[0., 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 1], [0, 1, 1, 0]])
    weights = torch.tensor([[1.] * 4] * 4) * adjacency

    g = graphrl.Graph(features, adjacency, weights)
    s = graphrl.State(g)

    s.add_node(0)
    s.add_node(2)
    s.add_node(3)
    s.add_node(1)

    s2 = graphrl.State(g)
    s2.add_node(0)

    assert s == s
    assert g == g
    assert s.substate_at_step(0) == graphrl.State(g)
    assert s.substate_at_step(1) == graphrl.State(g).add_node(0)
    assert s.substate_at_step(1) != graphrl.State(g).add_node(2)
    assert s.substate_at_step(2) == graphrl.State(g).add_node(0).add_node(2)
    i = 0
    for _ in s.substate_at_step(1):
        i += 1
    assert i == 1


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
