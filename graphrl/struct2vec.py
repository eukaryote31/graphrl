import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.init import xavier_normal_
from graphrl.graphrl import Graph, Solution
from typing import List


dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GraphEmbedder(nn.Module):
    def __init__(self, iters=3, embedsize=10):
        super(GraphEmbedder, self).__init__()
        self.iters = iters
        self.embedsize = embedsize

        self.w_selected = nn.Linear(1, embedsize)
        self.w_nbpriors = nn.Linear(embedsize, embedsize)
        self.w_nbweights = nn.Linear(embedsize, embedsize)
        self.w_nbweights_ew = nn.Linear(1, embedsize)
        self.w_q_reduc = nn.Linear(2 * embedsize, 1)
        self.w_q_allembed = nn.Linear(embedsize, embedsize)
        self.w_q_action = nn.Linear(embedsize, embedsize)

    def forward(self, sols: List[Solution]):
        graphsize = max([len(s.graph) for s in sols])
        batchsize = len(sols)
        embedsize = self.embedsize
        embeddings = torch.zeros(batchsize, graphsize, embedsize)

        weights = torch.stack([s.weights(graphsize) for s in sols])
        adjacency = torch.stack([s.adjacency(graphsize) for s in sols])
        features = torch.stack([s.features(graphsize) for s in sols])

        v_selected = self.w_selected(features.reshape(batchsize, -1, 1))
        weights = F.relu(self.w_nbweights_ew(weights.reshape(batchsize, graphsize, graphsize, 1)))
        v_weights = self.w_nbweights(torch.sum(weights, dim=1))

        # compute embeddings over iters
        for t in range(self.iters):
            v_priors = torch.bmm(adjacency, embeddings)
            v_priors = self.w_nbpriors(v_priors)
            newembeds = F.relu(v_selected + v_weights + v_priors)

            embeddings = newembeds

        sumembed = torch.sum(embeddings, dim=1)
        sumembed = self.w_q_allembed(sumembed)
        sumembed = sumembed.reshape(batchsize, 1, embedsize).expand(batchsize, graphsize, embedsize)
        peract = self.w_q_action(embeddings)
        q_vals = torch.cat((sumembed, peract), dim=2)
        q_vals = self.w_q_reduc(q_vals)[:, :, 0]

        return q_vals, embeddings
