import torch
import torch.nn as nn
import numpy as np
import ot
# from .ot_modules import compute_cost_mat
from otgnn.models import fused_gw_torch, compute_ot
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

class Penalty:
    def __init__(self, conn_eps = 1e-6):
        self.conn_eps = conn_eps
        self.__name__ = "Penalty terms"

    def conn_penalty(self, adjM):
        N = adjM.shape[0]
        device = adjM.device

        # get Laplacian
        L = torch.diag(torch.matmul(adjM, torch.ones(N, device = device))) - adjM
        L_mod = L + torch.ones_like(L, device = device)/N

        # calculate log dets
        return (-torch.logdet(L_mod + self.conn_eps * torch.eye(N, device = device)))

    def conn(self, prediction, target_batch):
        # Unpack inputs
        _, logits, scope = prediction
        _, _, bonds_logits = logits
        device = bonds_logits.device

        bonds_prob = nn.Softmax(dim = 1)(bonds_logits)[:,-1]# probability of no edge
        loss = torch.tensor(0., device=device) # reconstruction loss
        bond_idx = 0
        for _, num_atoms in scope:
            # Metric cost matrix for nodes
            pred_bonds = bonds_prob[bond_idx:bond_idx+num_atoms*num_atoms].view(num_atoms, num_atoms) # num_atoms^2 predictions for all the possible bonds

            loss += self.conn_penalty(pred_bonds)
            bond_idx += num_atoms * num_atoms
        return loss

    def __call__(self, prediction, target_batch):
        return self.conn(prediction, target_batch)
        

def compute_barycenter(pc_X, b_size, bary_pc_gain=1, num_iters=5):
    '''
    Computes the barycenter given fixed lambda weights.
    Params:
        W_lambda: [1, |D|], the vector of lambda to compute barycenter
        num_iters: (optional) Number of iterations to run
    '''
    # simplified form, just for our use case
    W_lambda = [1]
    pc_X_list = [pc_X]
    n_pc = len(W_lambda)
    pc_hidden = pc_X.shape[1]
    x_size = pc_X.shape[0]
    device = pc_X.device

    # Initialize barycenter: [b_size, pc_hidden]
    # torch.manual_seed(69)
    b_X = torch.empty([b_size, pc_hidden], device=device)
    nn.init.xavier_normal_(b_X, gain=bary_pc_gain)
    b_H = np.ones([b_size]) / b_size
    # compute the barycenter without gradients
    with torch.no_grad():
        for it in range(num_iters):
            # Initialize new barycenter: [pc_size, pc_hidden]
            b_X_new = torch.zeros([b_size, pc_hidden], device=device)
            all_ot_mats = []
            for pc_idx in range(n_pc):
                if W_lambda[pc_idx] == 0:
                    all_ot_mats.append(0) #dummy element
                    continue
                # pc_X: [pc_size, pc_hidden], pc_H: [pc_size, 1]
                pc_X = pc_X_list[pc_idx]
                pc_H = np.ones([x_size])/x_size
                # ot_mat = torch.tensor(
                #     b_H[:, None] * pc_H[None, :], device=self.args.device).float()
                # ot_mat: [pc_size, pc_size]
                _, _, ot_mat, _ = compute_ot(
                    X_1=b_X, X_2=pc_X, H_1=b_H, H_2=pc_H, device=device,
                    sinkhorn_entropy=0.1, sinkhorn_max_it= 500,
                    opt_method="emd")
                all_ot_mats.append(ot_mat)
                # b_X = \sum_i \lambda_i * T_i * pc_X_i
                b_X_new += W_lambda[pc_idx] * torch.matmul(ot_mat, pc_X)
            err = torch.norm(b_size * b_X_new - b_X)
            # print (err)
            if it >= 2 and err < 0.1: # Convergence criteria  TODO
                break
            b_X = b_size * b_X_new  ### TODO: add line search here ?

    b_X_list = []  # instead of updating in-place, concatenate and sum b_X
    for pc_idx in range(n_pc):
        if W_lambda[pc_idx] != 0:
            # print('pc[',pc_idx, '] / lam_[', pc_idx, '] = ', (self.pc_X_list[pc_idx]/ W_lambda[pc_idx]))
            b_X_list.append(W_lambda[pc_idx] * torch.matmul(all_ot_mats[pc_idx].float(), pc_X_list[pc_idx]))
    if len(b_X_list) == 0:
        print(W_lambda)
    b_X_new = b_size * torch.sum(torch.stack(b_X_list), dim=0)
    return b_X_new