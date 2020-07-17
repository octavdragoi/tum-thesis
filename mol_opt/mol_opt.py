# only to deal with importing things from above directory
import sys
sys.path.append("..") 

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from otgnn.models import GCN, compute_ot

class MolOpt(nn.Module):
    def __init__(self, args):
        """Create the model with all its components"""
        super(MolOpt, self).__init__()
        self.args = args

        # for embeddings
        self.GCN = GCN(self.args).to(device = args.device)
        self.ref = torch.randn(self.args.dim_tangent_space, self.args.pc_hidden, device = args.device)
        self.Href = np.ones(self.args.dim_tangent_space)/self.args.dim_tangent_space
        self.Nref = self.args.dim_tangent_space

        # for the optimizer part
        self.opt0 = nn.Linear(self.args.pc_hidden, self.args.n_hidden).to(device = args.device)
        self.opt1 = nn.Linear(self.args.n_hidden, self.args.pc_hidden).to(device = args.device)

    def encode(self, batch):
        # get GCN embedding
        embedding = self.GCN(batch)[0]

        return self.project(embedding, batch)

    def project(self, embedding, batch):
        """ Project on the tangent space
        As seen in the Kolouri paper and jupyter notebook 
        """
        tg_embedding = torch.empty(self.Nref * len(batch.scope), 50, device = self.args.device)
        for idx, (stx, lex) in enumerate(batch.scope):
            narrow = embedding.narrow(0, stx, lex)
            H = np.ones(lex)/lex
            _,_,OT_xy,_ = compute_ot(narrow, self.ref, H, self.Href, sinkhorn_entropy = 0.1, device = self.args.device)
            V = torch.matmul((self.Nref * OT_xy).T, narrow) - self.ref
            tg_embedding[idx*self.Nref : (idx+1)*self.Nref,:] = V / np.sqrt(self.Nref)
        
        return tg_embedding

    def optimize(self, x_embedding, x_batch):
        return self.opt1(nn.LeakyReLU()(self.opt0(x_embedding)))
        
    def forward(self, x_batch):
        x_embedding = self.encode(x_batch)
        x_delta_hat = self.optimize(x_embedding, x_batch)
        return x_embedding, x_delta_hat
