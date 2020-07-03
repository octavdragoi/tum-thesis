# only to deal with importing things from above directory
import sys
sys.path.append("..") 

import torch
import torch.nn as nn

from otgnn.models import GCN, compute_ot
from otgnn.datasets import PropDataset
from otgnn.graph import MolGraph

class MolOpt(nn.Module):
    def __init__(self, args):
        """Create the model with all its components"""
        super(MolOpt, self).__init__()
        self.args = args

        # for embeddings
        self.GCN = GCN(self.args)

        # for the optimizer part
        self.opt = nn.Linear(self.args.pc_hidden, self.args.pc_hidden)

        # loss
        self.delta_loss = nn.MSELoss()

    def encode(self, batch):
        return self.GCN(batch)

    def delta(self, x_embedding, y_embedding):
        return y_embedding - x_embedding

    def optimize(self, x_embedding):
        return self.opt(x_embedding)
        
    def forward(self, x_batch):
        x_embedding = self.encode(x_batch)
        x_delta_hat = self.optimize(x_embedding[0])
        return x_delta_hat

    def forward_train(self, x_batch, y_batch):
        x_embedding = self.encode(x_batch)
        x_delta_hat = self.optimize(x_embedding[0])

        # get the delta between the two nodes, to retrieve the embedding
        y_embedding = self.encode(y_batch)
        xy_delta = self.delta(y_embedding[0], x_embedding[0])

        return self.delta_loss(x_delta_hat, xy_delta)