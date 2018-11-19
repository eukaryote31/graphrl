import torch
import networkx as nx
from graphrl import struct2vec, graphrl, graphgen, trainer



dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test_trainer():
    dist = graphgen.MixedDistribution(sizerange=(40, 40))
    g = graphgen.MixedDistribution(sizerange=(100, 100)).sample_graph()
    emb = struct2vec.GraphEmbedder(embedsize=100, iters=4).to(dev)
    prob = graphrl.MVCProblem()
    optim = torch.optim.Adam(emb.parameters(), lr=0.0001)
    t = trainer.Trainer(dist, emb, prob, optim, batchsize=64, epsilon=1, updatedelay=2)
    for i in range(1000000):
        loss = t.train_ep()
        s = graphrl.Solution(g)

        while not prob.terminate(s):
            newvertex = s.pick_node(emb, 0)
            s = s.add_node(newvertex)

        print(i, ('%.5f' % t.epsilon), ('%.5f' % loss).rjust(15)
        , len(s))
        if i > 0 and t.epsilon > 0.05:
            t.epsilon -= 0.001

    assert 0


test_trainer()
#        , str(emb([graphrl.Solution(graphgen.ToyGraphDistribution().sample_graph()).add_node(1)])[0]).replace('\n', ''))
