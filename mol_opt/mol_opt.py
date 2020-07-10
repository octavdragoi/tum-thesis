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

        # for the optimizer part
        self.opt0 = nn.Linear(self.args.pc_hidden, self.args.n_hidden).to(device = args.device)
        self.opt1 = nn.Linear(self.args.n_hidden, self.args.pc_hidden).to(device = args.device)

    def encode(self, batch):
        return self.GCN(batch)[0]

    def optimize(self, x_embedding, x_batch):
        return self.opt1(nn.LeakyReLU()(self.opt0(x_embedding)))
        
    def forward(self, x_batch):
        x_embedding = self.encode(x_batch)
        x_delta_hat = self.optimize(x_embedding, x_batch)
        return x_embedding, x_delta_hat


    # implemented this to try and apply the error directly between the 
    # x and y embeddings. didn't work, though
    def forward_train(self, x_batch, y_batch):
        # get the predicted value
        x_embedding, x_delta_hat = self.forward(x_batch)

        # get the delta between the two nodes, to retrieve the embedding
        y_embedding = self.encode(y_batch)
        y_embedding_aligned = self.align(x_embedding, x_batch, y_embedding, y_batch)
        xy_delta = self.delta(y_embedding_aligned, x_embedding)

        # print (x_embedding.shape, y_embedding.shape, y_embedding_aligned.shape, xy_delta.shape, x_delta_hat.shape)

        return nn.MSELoss(x_delta_hat, xy_delta)

    def align(self, x_embedding, x_batch, y_embedding, y_batch):
        # using the OT permutation matrix between the two embeddings,
        # align them so that we can determine the delta vector
        # keep the original alignment of the x vector, of course
        # this is made tricker by the batched processing, though
        pmatrix = torch.zeros_like(y_embedding)
        for (stx, lex), (sty, ley) in zip(x_batch.scope, y_batch.scope):
            x_narrow = x_embedding.narrow(0, stx, lex)
            y_narrow = y_embedding.narrow(0, sty, ley)
            lenx = x_narrow.shape[0]
            leny = y_narrow.shape[0]
            Hx = np.ones(lenx)/lenx
            Hy = np.ones(leny)/leny

            OT_xy = compute_ot(x_narrow, y_narrow,
                opt_method = 'emd',
                sinkhorn_max_it = 150, device = self.args.device,
                H_1 = Hx, H_2 = Hy, sinkhorn_entropy = 0.1)

            pmatrix[stx:stx+lex,:] = torch.mm(OT_xy[2] * lenx, y_narrow)
        return pmatrix

    def delta(self, x_embedding, y_embedding):
        return y_embedding - x_embedding