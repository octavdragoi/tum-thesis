import os
import sys
# need the path to otgnn repo
sys.path.append(os.path.join(os.getcwd(), "otgnn")) 

from otgnn.models import GCN
from otgnn.graph import MolGraph
from otgnn.utils import save_model, load_model

from mol_opt.data_mol_opt import MolOptDataset
from mol_opt.data_mol_opt import get_loader
from mol_opt.arguments import get_args
from mol_opt.mol_opt import MolOpt

from rdkit.Chem import MolFromSmiles

import torch
import time

def get_latest_model(model_name, outdir):
    split_names = [x.split("_") for x in os.listdir(outdir)]
    try:
        max_epoch = max([int(x[2]) for x in split_names if x[0] == "model" and 
                            x[1] == "gcn" and len(x) == 3])
    except ValueError:
        print ("No model {} found in {}! Starting from scratch.".format(model_name, outdir))
        return None, 0
    return "{}_{}_{}".format("model", model_name, max_epoch), max_epoch

def main(args = None, molopt = None, data_loader = None):
    if args is None:
        args = get_args()
        args.output_dir = "mol_opt/output"

    # if model is not created yet, then create a new one
    prev_epoch = -1 
    if molopt is None:
        # preload previously trained model, if configured
        model_name, prev_epoch = get_latest_model(args.init_model, args.output_dir)
        if model_name is not None:
            molopt, config = load_model(os.path.join(args.output_dir, model_name), 
                MolOpt, args.device)
        else:
            molopt = MolOpt(args).to(device = args.device)

    # the data is from Wengong's repo
    if data_loader is None:
        data_loader = get_loader("iclr19-graph2graph/data/qed", 48, True)

    for epoch in range(prev_epoch + 1, prev_epoch + args.n_epochs + 1):
        # run the training procedure
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
        # compute the validation loss as well, at the end of the epoch?
        print("Epoch duration:", end - start)

        # save your progress along the way
        save_model(molopt, args, args.output_dir, "gcn_{}".format(epoch))
    
    return molopt

if __name__ == "__main__":
    main()