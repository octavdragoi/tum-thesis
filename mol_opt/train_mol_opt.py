import os
import sys
# need the path to otgnn repo
sys.path.append(os.path.join(os.getcwd(), "otgnn")) 

from otgnn.models import GCN
from otgnn.graph import MolGraph

from mol_opt.data_mol_opt import MolOptDataset
from mol_opt.data_mol_opt import get_loader
from mol_opt.arguments import get_args
from mol_opt import mol_opt

from rdkit.Chem import MolFromSmiles

import torch
import time

def main(molopt = None):
    sys.argv = [""]
    args = get_args()
    args.device = "cuda:0"

    # if model is not created yet, then create a new one
    if molopt is None:
        molopt = mol_opt.MolOpt(args)
        if args.cuda:
            molopt.cuda()

    # the data is from Wengong's repo
    data_loader = get_loader("iclr19-graph2graph/data/qed", 48, True)

    start = time.time()
    for idx, i in enumerate(data_loader):
        X = (MolGraph(i[0]))
        Y = (MolGraph(i[1]))
        
        # create your optimizer
        optimizer = torch.optim.SGD(molopt.parameters(), lr=0.01)

        # in your training loop:
        optimizer.zero_grad()   # zero the gradient buffers
        loss = molopt.forward_train(X, Y)
        loss.backward()
        optimizer.step()    # Does the update

        print (idx, loss)

    end = time.time()
    print(end - start)