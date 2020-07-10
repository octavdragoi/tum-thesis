import sys
sys.path.append("..") 

import torch
import torch.nn as nn
import torch.nn.functional as F

from otgnn.graph import SYMBOLS, FORMAL_CHARGES, BOND_TYPES

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

    def add_delta(self, x_embedding, xy_delta):
        """Get the new embedding from the old one, plus the error term""" 
        return (x_embedding + xy_delta)

    def forward(self, x_embedding, x_batch):
        """Predict symbols, charges, bonds logits independently"""
        symbols_logits = self.fc2_SYMBOLS(F.relu(self.fc1_SYMBOLS(x_embedding)))
        charges_logits = self.fc2_CHARGES(F.relu(self.fc1_CHARGES(x_embedding)))

        bonds_logits = torch.empty(0, n_BOND_TYPES, device=self.args.device)
        for stx, lex in x_batch.scope:
            x_narrow = x_embedding.narrow(0, stx, lex)
            x1 = x_narrow.view(lex, 1, -1).repeat(1, lex, 1)
            x2 = x_narrow.view(1, lex, -1).repeat(lex, 1, 1)
            _bonds = torch.cat((x1, x2), dim = 2)
            logits = self.fc2_BONDS(F.relu(self.fc1_BONDS(_bonds)))

            logits = logits.view(-1, n_BOND_TYPES)
            bonds_logits = torch.cat((bonds_logits, logits))

        return symbols_logits, charges_logits, bonds_logits

    def discretize(self, symbols_logits, charges_logits, bonds_logits):
        # discretize by taking logit argmax
        symbols_labels = torch.argmax(symbols_logits, dim=1)
        charges_labels = torch.argmax(charges_logits, dim=1)
        bonds_labels = torch.argmax(bonds_logits, dim=1)
        return symbols_labels, charges_labels, bonds_labels