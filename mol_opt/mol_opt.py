# only to deal with importing things from above directory
import sys
sys.path.append("..") 

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect

from otgnn.models import GCN, compute_ot

from mol_opt.transformer import make_model

class MolOpt(nn.Module):
    def __init__(self, args):
        """Create the model with all its components"""
        super(MolOpt, self).__init__()
        self.args = args

        # for embeddings
        self.GCN = GCN(self.args).to(device = args.device)

        # for the projection. ffn and transformer use this
        if self.args.model_type == "ffn" or self.args.model_type == "transformer" or self.args.model_type == "transformer-ae":
            self.ref = nn.parameter.Parameter(torch.randn(self.args.dim_tangent_space, self.args.pc_hidden, device = args.device))
            self.Href = np.ones(self.args.dim_tangent_space)/self.args.dim_tangent_space
            self.Nref = self.args.dim_tangent_space

        # for the optimizer part
        if self.args.model_type == "transformer":
            self.transformer = make_model(args)
        if self.args.model_type == "transformer-ae":
            self.max_num_atoms = args.max_num_atoms
            self.opt0 = nn.Linear(self.args.pc_hidden, self.args.n_hidden).to(device = args.device)
            self.opt1 = nn.Linear(self.args.n_hidden, self.args.pc_hidden).to(device = args.device)
        if self.args.model_type == "pointwise":
            self.opt0 = nn.Linear(self.args.pc_hidden, self.args.n_hidden).to(device = args.device)
            self.opt1 = nn.Linear(self.args.n_hidden, self.args.pc_hidden).to(device = args.device)
        if self.args.model_type == "ffn":
            self.opt0 = nn.Linear(self.Nref * self.args.pc_hidden, self.Nref * self.args.n_hidden).to(device = args.device)
            self.opt1 = nn.Linear(self.Nref * self.args.n_hidden, self.Nref * self.args.pc_hidden).to(device = args.device)
        if self.args.model_type == "molemb":
            self.max_num_atoms = args.max_num_atoms
            self.opt0 = nn.Linear(self.args.pc_hidden, self.args.n_hidden * self.max_num_atoms).to(device = args.device)
            self.opt1 = nn.Linear(self.args.n_hidden * self.max_num_atoms, self.args.pc_hidden * self.max_num_atoms).to(device = args.device)

        # only for models generating a molecule embedding
        if self.args.morgan_bits > 0:
            self.morg0 = nn.Linear(self.args.pc_hidden + self.args.morgan_bits, self.args.n_hidden).to(device = args.device)
            self.morg1 = nn.Linear(self.args.n_hidden, self.args.pc_hidden).to(device = args.device)

    def encode(self, batch):
        # get GCN embedding
        embedding = self.GCN(batch)[0]
        if self.args.model_type == "ffn" or self.args.model_type == "transformer":
            return self.project(embedding, batch)
        elif self.args.model_type == "slot" or self.args.model_type == "pointwise":
            return embedding
        elif self.args.model_type == "molemb" or self.args.model_type == "transformer-ae":
            if self.args.morgan_bits > 0:
                mol_embs = torch.zeros((self.args.batch_size, self.args.morgan_bits + self.args.pc_hidden), device = self.args.device)
            else:
                mol_embs = torch.zeros((self.args.batch_size, self.args.pc_hidden), device = self.args.device)
            for idx, (stx, lex) in enumerate(batch.scope):
                narrow = embedding.narrow(0, stx, lex)
                mol_embs[idx,:self.args.pc_hidden] = narrow.sum(axis = 0)
                if self.args.morgan_bits > 0:
                    mol_fingerp = GetMorganFingerprintAsBitVect(batch.rd_mols[0],2,nBits=self.args.morgan_bits)
                    mol_embs[idx,self.args.pc_hidden:][mol_fingerp] = 1
            return mol_embs

    def project(self, embedding, batch):
        """ Project on the tangent space
        As seen in the Kolouri paper and jupyter notebook 
        """
        tg_embedding = torch.empty(self.Nref * len(batch.scope), self.args.pc_hidden, device = self.args.device)
        for idx, (stx, lex) in enumerate(batch.scope):
            narrow = embedding.narrow(0, stx, lex)
            H = np.ones(lex)/lex
            _,_,OT_xy,_ = compute_ot(narrow, self.ref, H, self.Href, sinkhorn_entropy = 0.1, 
                    device = self.args.device, opt_method='emd', sinkhorn_max_it= 1000)
            V = torch.matmul((self.Nref * OT_xy).T, narrow) - self.ref
            tg_embedding[idx*self.Nref : (idx+1)*self.Nref,:] = V / np.sqrt(self.Nref)
        return tg_embedding

    def optimize(self, x_embedding, x_batch):
        if self.args.morgan_bits > 0:
            x_embedding = self.morg1(F.leaky_relu(self.morg0(x_embedding)))
        if self.args.model_type == "pointwise":
            return self.opt1(F.leaky_relu(self.opt0(x_embedding)))
        elif self.args.model_type == "ffn":
            x_embedding_view = x_embedding.view(-1, self.Nref * self.args.pc_hidden)
            yhat_embedding = self.opt1(F.leaky_relu(self.opt0(x_embedding_view)))
            return yhat_embedding.view(-1, self.args.pc_hidden)
        elif self.args.model_type == "transformer":
            return self.transformer(x_embedding, None)
        elif self.args.model_type == "transformer-ae": 
            return self.opt1(F.leaky_relu(self.opt0(x_embedding)))
        elif self.args.model_type == "slot":
            return x_embedding
        elif self.args.model_type == "molemb":
            return self.opt1(F.leaky_relu(self.opt0(x_embedding))).view(self.args.batch_size, self.args.max_num_atoms, self.args.pc_hidden)
        

    def forward(self, x_batch):
        x_encoding = self.encode(x_batch)
        x_embedding = self.optimize(x_encoding, x_batch)
        return x_encoding, x_embedding