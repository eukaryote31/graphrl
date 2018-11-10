import copy
import random
import collections
from bitarray import bitarray
import torch


class Problem:
    def cumul_reward(self, graph, state, fr, to):
        cumul = 0
        for i in range(fr, to):
            cumul += self.reward(graph, state.substate_at_step(i), state[i])
        return cumul


class MVCProblem(Problem):
    def terminate(self, graph, state):
        adjacency = graph.adjacency
        adjacency = adjacency.clone()
        for sel in state:
            adjacency[sel, :].fill_(0)
            adjacency[:, sel].fill_(0)
        return not torch.any(torch.eq(adjacency, 1))

    def reward(self, graph, state, action):
        return -1


class Graph:
    def __init__(self, features, adjacency, weights):
        assert features.size(0) == adjacency.size(0)
        assert adjacency.size(0) == adjacency.size(1)
        assert adjacency.size() == weights.size()
        self.features = features
        self.adjacency = adjacency
        self.weights = weights

    def adjacent_nodes(self, node):
        return self.adjacency[node]

    def node_weight(self, node):
        return self.weights[node]

    def node_features(self, node):
        return self.features[node]

    def __len__(self):
        return self.features.size(0)

    def __eq__(self, val):
        return torch.all(torch.eq(self.features, val.features)) \
           and torch.all(torch.eq(self.adjacency, val.adjacency)) \
           and torch.all(torch.eq(self.weights, val.weights))


class State:
    def __init__(self, graph, state=None, inv_state=None, n_steps=None):
        self.graph = graph
        self.state = state if state else []
        self.nsteps = n_steps
        self.inv_state = inv_state
        if not inv_state:
            self.inv_state = bitarray(len(graph))
            for i in range(len(graph)):
                self.inv_state[i] = 1
            for i in self.state:
                self.inv_state[i] = 0

    def add_node(self, nodeidx):
        self._resolve_view()
        self.state.append(nodeidx)
        self.inv_state[nodeidx] = 0
        return self

    def nodes_included(self):
        if self.nsteps is not None:
            return self.state[:self.nsteps]
        return self.state

    def pick_random_node(self):
        self._resolve_view()
        num_nodes = len(self.graph) - len(self.state)
        if num_nodes <= 0:
            raise IndexError()
        pos = random.randrange(0, num_nodes)
        for i, e in enumerate(self.inv_state):
            if e:
                if pos == 0:
                    return i
                else:
                    pos -= 1

        raise Error('inconsistent state!')

    def __contains__(self, node):
        return not self.inv_state[node]

    def __len__(self):
        if self.nsteps is not None:
            return self.nsteps
        else:
            return len(self.state)

    def __iter__(self):
        return StateIter(self)

    def __str__(self):
        return str(self.state[:self.nsteps] if self.nsteps else self.state)

    def _resolve_view(self):
        if self.nsteps is not None:
            # resolve view by making copy of data
            self.inv_state = self.inv_state.copy()

            for i in self.state[self.nsteps:]:
                self.inv_state[i] = 1
            self.state = self.state[:self.nsteps]
            self.nsteps = 0

    def __getitem__(self, key):
        if self.nsteps is not None:
            if key >= self.nsteps:
                raise IndexError
        return self.state[key]

    def __eq__(self, val):
        try:
            if len(self) != len(val):
                return False

            if self.graph != val.graph:
                return False

            if self.state is val.state and self.nsteps == val.nsteps:
                return True

            for i in range(len(self)):
                if self[i] != val[i]:
                    return False

            return True
        except TypeError:
            return False

    def substate_at_step(self, step):
        return State(self.graph, self.state, self.inv_state, step)


class StateIter:
    def __init__(self, state):
        self.state = state
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.i < len(self.state):
            r = self.state.state[self.i]
            self.i += 1
            return r
        else:
            raise StopIteration()

    def next(self):
        return self.__next__()


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.arr = []
        self.index = 0

    def add(self, state):
        if len(self.arr) < self.capacity:
            self.arr.append(state)
            self.index = len(self.arr) % self.capacity
        else:
            self.arr[self.index] = state
            self.index = (self.index + 1) % self.capacity

    def sample(self, batchsize):
        assert batchsize <= len(self.arr)

        return random.sample(self.arr, batchsize)

    def __len__(self):
        return len(self.arr)

    def __iter__(self):
        return self.arr.__iter__()

    def __str__(self):
        return self.arr.__str__()

    def __eq__(self, val):
        return self.arr == val.arr
