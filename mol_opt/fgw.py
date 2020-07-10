import torch
import torch.nn as nn
import numpy as np
import ot
# from .ot_modules import compute_cost_mat
from otgnn.models import fused_gw_torch
from otgnn.graph import SYMBOLS, FORMAL_CHARGES, BOND_TYPES, get_bt_index

def encode_target(y_batch, device = 'cpu'):
    # get encoding of symbols
    symbols_len = sum([x[1] for x in y_batch.scope])
    bonds_len = sum([x[1] ** 2 for x in y_batch.scope])

    symbols_lst = [ys.symbol for y in y_batch.mols for ys in y.atoms]
    target_symbols = torch.zeros((symbols_len, len(SYMBOLS)), device = device)
    for idxl, l in enumerate(symbols_lst):
        for idxs, s in enumerate(SYMBOLS):
            if (s == l):
                symbols_lst[idxl] = idxs
                target_symbols[idxl, idxs] = 1

    target_bonds = torch.zeros(bonds_len, len(BOND_TYPES), device = device)
    target_bonds[:,-1].fill_(1)

    # encode bonds
    bond_idx = 0
    for mol_idx, mol in enumerate(y_batch.mols):
        lex = len(mol.atoms)
        for x in mol.bonds:
            # in this encoding, in_atom is the first one, out_atom is the second one
            curr_bond_idx = bond_idx + lex * x.in_atom_idx + x.out_atom_idx
            target_bonds[curr_bond_idx, -1] = 0
            target_bonds[curr_bond_idx, get_bt_index(x.bond_type)] = 1

        bond_idx += lex ** 2

    return target_symbols, target_bonds

class FGW:
    def __init__(self, *, alpha, include_charge=True):
        self.alpha = alpha
        self.include_charge = include_charge
        self.__name__ = 'FGW'

    def __call__(self, prediction, target_batch):
        # Unpack inputs
        labels, logits, scope = prediction
        # symbols_labels, charges_labels, bonds_labels = labels
        symbols_logits, charges_logits, bonds_logits = logits
        symbols_nll, charges_nll, bonds_nll = -nn.LogSoftmax(dim=1)(symbols_logits), -nn.LogSoftmax(dim=1)(charges_logits), -nn.LogSoftmax(dim=1)(bonds_logits)
        device = symbols_logits.device
        # target[which mol][which feature] -> n_atoms/n_bonds x len(feature)

        # encode the molecule in the required way
        target_symbols, target_bonds = encode_target(target_batch, device = device)

        # Compute FGW per molecule in batch and add together
        loss = torch.tensor(0., device=device) # reconstruction loss
        bond_idx = 0
        for mol_idx, (st, num_atoms) in enumerate(scope):
            # Metric cost matrix for nodes
            pred_symbols_nll = symbols_nll[st:st+num_atoms]
            target_symbols_rescaled = target_symbols[st:st+num_atoms]
            M = target_symbols_rescaled.matmul(pred_symbols_nll.transpose(0, 1))
            # M = compute_cost_mat(target_symbols_rescaled, pred_symbols_nll, False, 'dot')

            # Whether to ignore predicted charges
            # if self.include_charge:
            #     pred_charges_nll = charges_nll[st:st+num_atoms]
            #     target_charges_rescaled = target[mol_idx]['FORMAL_CHARGES']
            #     M += target_charges_rescaled.matmul(pred_charges_nll.transpose(0, 1))
                # M += compute_cost_mat(target_charges_rescaled, pred_charges_nll, False, 'dot')

            # Metric cost matrix for edges
            # IMPORTANT: this is unbalanced? Especially if scaled
            pred_bonds_nll = bonds_nll[bond_idx:bond_idx+num_atoms*num_atoms].view(num_atoms, num_atoms, -1) # num_atoms^2 predictions for all the possible bonds
            target_bonds_rescaled = target_bonds[bond_idx:bond_idx+num_atoms*num_atoms].view(num_atoms, num_atoms, -1) # num_atoms^2 predictions for all the possible bonds

            atom_gw_dist, bond_gw_dist, _, _ = fused_gw_torch(
                M=M,
                C1= target_bonds_rescaled,
                C2= -1 * pred_bonds_nll,
                p1=np.ones([num_atoms]) / float(num_atoms),
                p2=np.ones([num_atoms]) / float(num_atoms),
                dist_type='dot',
                nce_reg = True,
                alpha=self.alpha,
                device=device
            )
            loss += atom_gw_dist + bond_gw_dist
            bond_idx += num_atoms * num_atoms
        return loss
