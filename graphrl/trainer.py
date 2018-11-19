import torch
from graphrl import graphgen, graphrl, struct2vec


class Trainer:
    def __init__(self,
                 distribution: graphgen.Distribution,
                 embedder: struct2vec.GraphEmbedder,
                 problem: graphrl.Problem,
                 optimizer: torch.optim.Optimizer,
                 replaycapacity=10000,
                 updatedelay=5,
                 batchsize=128,
                 epsilon=0.75
                 ):
        self.distribution = distribution
        self.embedder = embedder
        self.problem = problem
        self.rmem = graphrl.ReplayMemory(replaycapacity)
        self.updatedelay = updatedelay
        self.batchsize = batchsize
        self.optimizer = optimizer
        self.epsilon = epsilon

    def train_ep(self):
        graph = self.distribution.sample_graph()
        sol = graphrl.Solution(graph)
        time = 0
        loss = 0
        while not self.problem.terminate(sol):
            newvertex = sol.pick_node(self.embedder, self.epsilon)
            sol = sol.add_node(newvertex)

            if time >= self.updatedelay:
                self.rmem.add(struct2vec.TrainingCase(sol, self.problem, self.updatedelay))

                if len(self.rmem) >= self.batchsize:
                    training = self.rmem.sample(self.batchsize)
                    loss += self.embedder.train(training, self.problem, (-100, 0))
                    self.optimizer.step()
            time += 1
        return loss
