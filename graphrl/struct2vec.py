import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.init import xavier_normal_
from graphrl.graphrl import Graph, Solution, Problem
from typing import List


class TrainingCase:
    def __init__(self,
                 solution: Solution,
                 problem: Problem,
                 updatedelay
                 ):
        assert len(solution) >= updatedelay
        assert updatedelay > 0

        t1 = len(solution)
        t0 = t1 - updatedelay

        self.solution0 = solution.subsolution_at_step(t0)
        self.nextvertex = solution[t0]
        self.cumulreward = problem.cumul_reward(solution, t0, t1)
        self.solution1 = solution


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)


class GraphEmbedder(nn.Module):
    def __init__(self,
                 device='cpu',
                 iters=5,
                 embedsize=10,
                 discountfactor=1,
                 ):
        super(GraphEmbedder, self).__init__()
        self.iters = iters
        self.embedsize = embedsize
        self.discountfactor = discountfactor

        self.w_selected = nn.Linear(1, embedsize, bias=False)
        self.w_nbpriors = nn.Linear(embedsize, embedsize, bias=False)
        self.w_nbweights = nn.Linear(embedsize, embedsize, bias=False)
        self.w_nbweights_ew = nn.Linear(1, embedsize, bias=False)
        self.w_q_reduc = nn.Linear(2 * embedsize, 1, bias=False)
        self.w_q_allembed = nn.Linear(embedsize, embedsize, bias=False)
#        self.w_q_allembed2 = nn.Linear(embedsize, embedsize, bias=False)
        self.w_q_action = nn.Linear(embedsize, embedsize, bias=False)
#        self.w_q_action2 = nn.Linear(embedsize, embedsize, bias=False)

        self.dev = device

        self.apply(init_weights)

    def forward(self, sols: List[Solution]):
        graphsize = max([len(s.graph) for s in sols])
        batchsize = len(sols)
        embedsize = self.embedsize
        embeddings = torch.zeros(batchsize, graphsize,
                                 embedsize, device=self.dev)

        weights = torch.stack([s.weights(graphsize) for s in sols]).to(self.dev)
        adjacency = torch.stack([s.adjacency(graphsize) for s in sols]).to(self.dev)
        features = torch.stack([s.features(graphsize) for s in sols]).to(self.dev)

        v_selected = self.w_selected(features.reshape(batchsize, -1, 1))
        weights = F.relu(self.w_nbweights_ew(
            weights.reshape(batchsize, graphsize, graphsize, 1)))
        v_weights = self.w_nbweights(torch.sum(weights, dim=1))

        # compute embeddings over iters
        for t in range(self.iters):
            v_priors = torch.bmm(adjacency, embeddings)
            v_priors = self.w_nbpriors(v_priors)
            newembeds = F.relu(v_selected + v_weights + v_priors)

            embeddings = newembeds

        sumembed = torch.sum(embeddings, dim=1)
        sumembed = self.w_q_allembed(sumembed)
#        sumembed = F.relu(sumembed)
#        sumembed = self.w_q_allembed2(sumembed)
        sumembed = sumembed.reshape(batchsize, 1, embedsize).expand(
            batchsize, graphsize, embedsize)
        peract = self.w_q_action(embeddings)
#        peract = F.relu(peract)
#        peract = self.w_q_action2(peract)
        q_vals = torch.cat((sumembed, peract), dim=2)
        q_vals = self.w_q_reduc(q_vals)[:, :, 0]

        return q_vals, embeddings

    def train(self, cases: List[TrainingCase], problem: Problem, q_clamp: tuple):
        self.zero_grad()
        batchsize = len(cases)
        sols0 = [x.solution0 for x in cases]
        sols1 = [x.solution1 for x in cases]
        verts0 = [x.nextvertex for x in cases]
        qvs, _ = self(sols0 + sols1)
        qvs0, qvs1 = qvs[:batchsize], qvs[batchsize:]
        qvs1 = qvs1.detach()

#        print((qvs0 - qvs1).mean())

        expectedq1, _ = qvs1.max(dim=1)

        # dont grad future Q-val
#        expectedq1 = expectedq1.detach()

        # discount future expected reward
        expectedq1 *= self.discountfactor

        er = [problem.cumul_reward(s1, len(s0), len(s1)) for s0, s1 in zip(sols0, sols1)]
        empiricalreward = torch.tensor(er, dtype=torch.get_default_dtype(), device=self.dev)

        expectedq1 = expectedq1.clamp(*q_clamp)
        target = empiricalreward + expectedq1
#        print(empiricalreward)

        qvs0 = qvs0[torch.arange(batchsize), verts0]
#        print(empiricalreward)

        loss = F.mse_loss(qvs0, target)

#        print(('%.5f' % (loss, )).rjust(15), qvs1.mean())

        loss.backward()

        return loss
