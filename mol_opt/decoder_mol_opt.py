import sys
sys.path.append("..") 

import torch
import torch.nn as nn
import torch.nn.functional as F

from otgnn.graph import SYMBOLS, FORMAL_CHARGES, BOND_TYPES
from mol_opt.ot_utils import compute_barycenter

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

        # TODO: Make a ModuleDict from feature to layers
        self.fc1_SYMBOLS = nn.Linear(self.args.pc_hidden, self.args.pred_hidden).to(device = args.device)
        self.fc2_SYMBOLS = nn.Linear(self.args.pred_hidden, n_SYMBOLS).to(device = args.device)
        self.fc1_CHARGES = nn.Linear(self.args.pc_hidden, self.args.pred_hidden).to(device = args.device)
        self.fc2_CHARGES = nn.Linear(self.args.pred_hidden, n_FORMAL_CHARGES).to(device = args.device)
        self.fc1_BONDS = nn.Linear(self.args.pc_hidden + self.args.pc_hidden, self.args.pred_hidden).to(device = args.device) # input latent representation of both atoms combined
        self.fc2_BONDS = nn.Linear(self.args.pred_hidden, n_BOND_TYPES).to(device = args.device)

        self.Nref = self.args.dim_tangent_space

    def add_delta(self, x_embedding, xy_delta):
        """Get the new embedding from the old one, plus the error term""" 
        return (x_embedding + xy_delta)

    def forward(self, x_embedding, x_batch):
        """Predict symbols, charges, bonds logits independently"""
        bonds_logits = torch.empty(0, n_BOND_TYPES, device=self.args.device)
        symbols_logits = torch.empty(0, n_SYMBOLS, device=self.args.device)
        charges_logits = torch.empty(0, n_FORMAL_CHARGES, device=self.args.device)

        for idx, (stx, lex) in enumerate(x_batch.scope):
            # x_narrow = x_embedding.narrow(0, stx, lex)
            x_narrow = x_embedding.narrow(0, idx*self.Nref, self.Nref)
            # cheating a bit here, by looking at what # of atoms should be
            x_narrow_resized = compute_barycenter(x_narrow, lex)

            symbols_logits_mol = self.fc2_SYMBOLS(F.relu(self.fc1_SYMBOLS(x_narrow_resized)))
            symbols_logits_mol = symbols_logits_mol.view(-1, n_SYMBOLS)
            symbols_logits = torch.cat((symbols_logits, symbols_logits_mol))

            charges_logits_mol = self.fc2_CHARGES(F.relu(self.fc1_CHARGES(x_narrow_resized)))
            charges_logits_mol = charges_logits_mol.view(-1, n_FORMAL_CHARGES)
            charges_logits = torch.cat((charges_logits, charges_logits_mol))

            x1 = x_narrow_resized.view(lex, 1, -1).repeat(1, lex, 1)
            x2 = x_narrow_resized.view(1, lex, -1).repeat(lex, 1, 1)
            _bonds = torch.cat((x1, x2), dim = 2)
            bonds_logits_mol = self.fc2_BONDS(F.relu(self.fc1_BONDS(_bonds)))
            # add matrix with its transpose, to get the 
            bonds_logits_mol = bonds_logits_mol.view(lex, lex, n_BOND_TYPES) 
            bonds_logits_mol = bonds_logits_mol + bonds_logits_mol.permute(1,0,2) 
            bonds_logits_mol = bonds_logits_mol.view(-1, n_BOND_TYPES)
            bonds_logits = torch.cat((bonds_logits, bonds_logits_mol))

        return symbols_logits, charges_logits, bonds_logits

    def discretize(self, symbols_logits, charges_logits, bonds_logits):
        # discretize by taking logit argmax
        symbols_labels = torch.argmax(symbols_logits, dim=1)
        charges_labels = torch.argmax(charges_logits, dim=1)
        bonds_labels = torch.argmax(bonds_logits, dim=1)
        return symbols_labels, charges_labels, bonds_labels