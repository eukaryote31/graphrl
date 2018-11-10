import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.init import xavier_normal_


dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GraphEmbedder(nn.Module):
    def __init__(self, iters: int, embedsize: int):
        super(GraphEmbedder, self).__init__()
        self.iters = iters
        self.embedsize = embedsize

        self.w_selected = nn.Parameter(torch.zeros(embedsize).normal_(0, 1))
        self.w_nbpriors = nn.Parameter(torch.zeros(embedsize, embedsize).normal_(0, 1))
        self.w_nbweights = nn.Parameter(torch.zeros(embedsize, embedsize).normal_(0, 1))
        self.w_nbweights_ew = nn.Parameter(torch.zeros(embedsize).normal_(0, 1))
        self.w_q_reduc = nn.Parameter(torch.zeros(1, 2 * embedsize).normal_(0, 1))
        self.w_q_allembed = nn.Parameter(torch.zeros(embedsize, embedsize).normal_(0, 1))
        self.w_q_action = nn.Parameter(torch.zeros(embedsize, embedsize).normal_(0, 1))

    def forward(self, graph: tuple):
        features, adjacency, weights = graph

        embedsize = self.embedsize
        embeddings = torch.zeros(len(features), embedsize)

        # compute embeddings for each node
        for t in range(self.iters):
            newembeds = torch.zeros(len(features), embedsize)

            for node, nbrs in enumerate(adjacency):
                is_selected = torch.tensor(1. if features[node] else 0.).to(dev)

                sum_nb = torch.zeros(embedsize)
                sum_weights = torch.zeros(embedsize)

                for nbr in nbrs:
                    weight = torch.tensor(float(weights[node][nbr])).to(dev)
                    prevembed = embeddings[nbr]

                    sum_nb += prevembed
                    sum_weights += nn.functional.leaky_relu(self.w_nbweights_ew * weight)

                sum_nb = torch.mv(self.w_nbpriors, sum_nb)
                sum_weights = torch.mv(self.w_nbweights, sum_weights)

                emb = nn.functional.leaky_relu(self.w_selected * is_selected +
                                         sum_nb +
                                         sum_weights)
                newembeds[node] = emb
            embeddings = newembeds

        sumembed = torch.sum(embeddings, dim=0)
        sumembed = torch.mv(self.w_q_allembed, sumembed)

        q_vals = []
        for i in range(len(features)):
            value = torch.cat([sumembed, torch.mv(self.w_q_action, embeddings[i])])
            value = nn.functional.leaky_relu(value)
            value = torch.mv(self.w_q_reduc, value)
            q_vals.append(value)

        return q_vals, embeddings
