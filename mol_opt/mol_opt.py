# only to deal with importing things from above directory
import sys
sys.path.append("..") 

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from otgnn.models import GCN, compute_ot

class MolOpt(nn.Module):
    def __init__(self, args):
        """Create the model with all its components"""
        super(MolOpt, self).__init__()
        self.args = args

        # for embeddings
        self.GCN = GCN(self.args).to(device = args.device)

    def encode(self, batch):
        # get GCN embedding
        embedding = self.GCN(batch)[0]

        return embedding

    def forward(self, x_batch):
        x_embedding = self.encode(x_batch)
        return x_embedding
