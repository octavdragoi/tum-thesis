import sys
sys.path.append("..") 

import torch
import torch.nn as nn
import torch.nn.functional as F
from slot_attention import SlotAttention
from torch.autograd import Variable

from otgnn.graph import SYMBOLS, FORMAL_CHARGES, BOND_TYPES
from mol_opt.ot_utils import compute_barycenter
from mol_opt.transformer import make_model

n_SYMBOLS = len(SYMBOLS)
n_FORMAL_CHARGES = len(FORMAL_CHARGES)
n_BOND_TYPES = len(BOND_TYPES)

class MolOptDecoder(nn.Module):
    def __init__(self, args):
        super(MolOptDecoder, self).__init__()
        self.args = args

        # self.n_SYMBOLS = args['output_dims']['SYMBOLS']
        # self.n_FORMAL_CHARGES = args['output_dims']['FORMAL_CHARGES']
        # self.n_BOND_TYPES = args['output_dims']['BOND_TYPES']

        # create the slot attention module
        if args.model_type == "slot":
            self.slot_att = SlotAttention(num_slots = 30, dim=args.pc_hidden, iters = 3)

        if self.args.model_type == "ffn" or args.model_type == "transformer":
            self.Nref = self.args.dim_tangent_space

        if self.args.model_type == "molemb" or args.model_type == "transformer-ae":
            self.max_num_atoms = self.args.max_num_atoms

        if self.args.model_type == "transformer-ae" or args.model_type == "transformer":
            self.transformer = make_model(args).to(device = args.device)

        # TODO: Make a ModuleDict from feature to layers
        self.fc1_SYMBOLS = nn.Linear(self.args.pc_hidden, self.args.pred_hidden).to(device = args.device)
        self.fc2_SYMBOLS = nn.Linear(self.args.pred_hidden, n_SYMBOLS).to(device = args.device)
        self.fc1_CHARGES = nn.Linear(self.args.pc_hidden, self.args.pred_hidden).to(device = args.device)
        self.fc2_CHARGES = nn.Linear(self.args.pred_hidden, n_FORMAL_CHARGES).to(device = args.device)
        self.fc1_BONDS = nn.Linear(2 * self.args.pc_hidden, 2 * self.args.pred_hidden).to(device = args.device) # input latent representation of both atoms combined
        self.fc2_BONDS = nn.Linear(2 * self.args.pred_hidden, n_BOND_TYPES).to(device = args.device)

    def forward(self, x_embedding, x_batch, y_batch):
        """Predict symbols, charges, bonds logits independently"""
        bonds_logits = torch.empty(0, n_BOND_TYPES, device=self.args.device)
        symbols_logits = torch.empty(0, n_SYMBOLS, device=self.args.device)
        charges_logits = torch.empty(0, n_FORMAL_CHARGES, device=self.args.device)

        for idx, (stx, lex) in enumerate(x_batch.scope):
            # print (stx, lex)
            _, ley = y_batch.scope[idx]

            # cheating a bit here, by looking at what # of atoms should be
            if self.args.model_type == "pointwise" or self.args.model_type == "deepsets":
                # if lex != ley:
                #     raise RuntimeError("{}!={}, which is required for pointwise optimization".\
                #         format(lex, ley))
                x_narrow = x_embedding[stx:stx+lex]
                if lex != ley:
                    yhat_narrow = compute_barycenter(x_narrow, ley).unsqueeze(0)
                else:
                    yhat_narrow = x_narrow.unsqueeze(0)
            if self.args.model_type == "slot":
                x_narrow = x_embedding[stx:stx+lex].unsqueeze(0)
                yhat_narrow = self.slot_att(x_narrow, num_slots = ley)
            elif self.args.model_type == "ffn":
                x_narrow = x_embedding[idx*self.Nref:(idx+1)*self.Nref]
                yhat_narrow = compute_barycenter(x_narrow, ley).unsqueeze(0)
            elif self.args.model_type == "molemb":
                yhat_narrow = x_embedding[idx,:ley].unsqueeze(0)
            elif self.args.model_type == "transformer-ae"or self.args.model_type == "transformer":
                ys = torch.zeros((1,self.args.pc_hidden),device = self.args.device)
                for i in range(ley+1):
                    # print (i, ys.shape, x_embedding[i].shape)
                    out = self.transformer(x_embedding[i], Variable(ys), None, None)
                    ys = torch.cat([ys, out[-1].unsqueeze(0)])
                yhat_narrow = out[1:].unsqueeze(0)
                # print (yhat_narrow.shape)

            
            # print (yhat_narrow.shape)

            symbols_logits_mol = self.fc2_SYMBOLS(F.leaky_relu(self.fc1_SYMBOLS(yhat_narrow)))
            symbols_logits_mol = symbols_logits_mol.view(-1, n_SYMBOLS)
            symbols_logits = torch.cat((symbols_logits, symbols_logits_mol))

            charges_logits_mol = self.fc2_CHARGES(F.leaky_relu(self.fc1_CHARGES(yhat_narrow)))
            charges_logits_mol = charges_logits_mol.view(-1, n_FORMAL_CHARGES)
            charges_logits = torch.cat((charges_logits, charges_logits_mol))

            x1 = yhat_narrow.view(ley, 1, -1).repeat(1, ley, 1)
            x2 = yhat_narrow.view(1, ley, -1).repeat(ley, 1, 1)
            # print(x1.shape)
            # print(x2.shape)
            _bonds = torch.cat((x1, x2), dim = 2)
            # print (_bonds.shape)
            bonds_logits_mol = self.fc2_BONDS(F.leaky_relu(self.fc1_BONDS(_bonds)))
            # add matrix with its transpose, to get a symmetric matrix 
            bonds_logits_mol = bonds_logits_mol.view(ley, ley, n_BOND_TYPES) 
            bonds_logits_mol = bonds_logits_mol + bonds_logits_mol.permute(1,0,2) 
            # fix diagonal entries, always predict no bond on the diagonal
            bonds_logits_mol.diagonal()[0:4] = -1e3 * torch.ones_like(bonds_logits_mol.diagonal()[0:4])
            bonds_logits_mol.diagonal()[4] =  1e3 * torch.ones_like(bonds_logits_mol.diagonal()[4])
            bonds_logits_mol = bonds_logits_mol.view(-1, n_BOND_TYPES)
            bonds_logits = torch.cat((bonds_logits, bonds_logits_mol))

        return symbols_logits, charges_logits, bonds_logits

    def discretize_argmax(self, symbols_logits, charges_logits, bonds_logits):
        # discretize by taking logit argmax
        symbols_labels = torch.argmax(symbols_logits, dim=1)
        charges_labels = torch.argmax(charges_logits, dim=1)
        bonds_labels = torch.argmax(bonds_logits, dim=1)
        return symbols_labels, charges_labels, bonds_labels

    def discretize_gumbel(self, symbols_logits, charges_logits, bonds_logits, tau = 1):
        # discretize by taking logit argmax
        symbols_labels = torch.argmax(F.gumbel_softmax(symbols_logits, dim=1, tau=tau), dim = 1)
        charges_labels = torch.argmax(F.gumbel_softmax(charges_logits, dim=1, tau=tau), dim = 1)
        bonds_labels = torch.argmax(F.gumbel_softmax(bonds_logits, dim=1, tau=tau), dim = 1)
        return symbols_labels, charges_labels, bonds_labels
