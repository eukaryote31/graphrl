import torch
from graphrl import graphgen, graphrl, struct2vec


class Trainer:
    def __init__(self,
                 distribution: graphgen.Distribution,
                 embedder: struct2vec.GraphEmbedder,
                 problem: graphrl.Problem,
                 optimizer: torch.optim.Optimizer,
                 device: torch.device,
                 replaycapacity=10000,
                 updatedelay=5,
                 batchsize=128,
                 epsilon=0.75,
                 q_clamp=(-100, 0)
                 ):
        self.distribution = distribution
        self.problem = problem
        self.rmem = graphrl.ReplayMemory(replaycapacity)
        self.updatedelay = updatedelay
        self.batchsize = batchsize
        self.optimizer = optimizer
        self.epsilon = epsilon
        self.q_clamp = q_clamp
        self.dev = device
        self.embedder = embedder.to(device)

    def train_ep(self):
        graph = self.distribution.sample_graph()
        sol = graphrl.Solution(graph, device=self.dev)
        time = 0
        loss = 0
        grad = 0
        while not self.problem.terminate(sol):
            newvertex = sol.pick_node(self.embedder, self.epsilon)
            sol = sol.add_node(newvertex)

            if time >= self.updatedelay:
                self.rmem.add(struct2vec.TrainingCase(sol, self.problem, self.updatedelay))

                if len(self.rmem) >= self.batchsize:
                    training = self.rmem.sample(self.batchsize)
                    loss += self.embedder.train(training, self.problem, self.q_clamp)
                    gnorm = torch.nn.utils.clip_grad_norm_(self.embedder.parameters(), 10)
                    grad += gnorm
#                    total_grad = torch.nn.utils.clip_grad_norm_(self.embedder.parameters(), 1)
#                    print(total_grad)
                    self.optimizer.step()
            time += 1
        grad /= time
        return loss, grad
