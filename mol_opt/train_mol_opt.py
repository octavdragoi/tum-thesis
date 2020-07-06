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

def main(args = None, molopt = None, data_loader = None):
    if args is None:
        args = get_args()

    # if model is not created yet, then create a new one
    if molopt is None:
        molopt = mol_opt.MolOpt(args).to(device = args.device)

    # the data is from Wengong's repo
    if data_loader is None:
        data_loader = get_loader("iclr19-graph2graph/data/qed", 48, True)

    for epoch in range(args.n_epochs):
        start = time.time()
        print ("Epoch:", epoch)
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

            print ("Iter: {}, loss: {}".format(idx, loss.item()))

        end = time.time()
        print("Epoch duration:", end - start)
    
    return molopt

if __name__ == "__main__":
    main()