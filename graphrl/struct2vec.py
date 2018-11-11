import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.init import xavier_normal_
from graphrl.graphrl import Graph


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

    def forward(self, graph: Graph):
        embedsize = self.embedsize
        embeddings = torch.zeros(len(graph), embedsize)

        v_selected = self.w_selected(graph.features.reshape(-1, 1))
        weights = F.relu(self.w_nbweights_ew(graph.weights.reshape(-1, 1)))
        v_weights = self.w_nbweights(torch.sum(weights, dim=0))
        # compute embeddings over iters
        for t in range(self.iters):
            v_priors = torch.mm(graph.adjacency, embeddings)
            v_priors = self.w_nbpriors(v_priors)

            newembeds = F.relu(v_selected + v_weights + v_priors)

            embeddings = newembeds

        sumembed = torch.sum(embeddings, dim=0)
        sumembed = self.w_q_allembed(sumembed)
        sumembed = sumembed.reshape(1, embedsize).expand(len(graph), embedsize)
        peract = self.w_q_action(embeddings)
        q_vals = torch.cat((sumembed, peract), dim=1)
        q_vals = self.w_q_reduc(q_vals)[:, 0]

        return q_vals, embeddings
