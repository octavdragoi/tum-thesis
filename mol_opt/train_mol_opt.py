import os
import sys
# need the path to otgnn repo
sys.path.append(os.path.join(os.getcwd(), "otgnn")) 

from otgnn.models import GCN
from otgnn.graph import MolGraph
from otgnn.utils import save_model, load_model, StatsTracker

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
                            x[1] == model_name and len(x) == 3])
    except ValueError:
        print ("No model {} found in {}! Starting from scratch.".format(model_name, outdir))
        return None, 0
    return "{}_{}_{}".format("model", model_name, max_epoch), max_epoch


def main(args = None, molopt = None, train_data_loader = None, val_data_loader = None):
    if args is None:
        args = get_args()
        args.output_dir = "mol_opt/output"

    # if model is not created yet, then create a new one
    prev_epoch = -1 
    if molopt is None:
        # preload previously trained model, if configured
        model_name, prev_epoch = get_latest_model(args.init_model, args.output_dir)
        if model_name is not None:
            molopt, _ = load_model(os.path.join(args.output_dir, model_name), 
                MolOpt, args.device)
        else:
            molopt = MolOpt(args).to(device = args.device)

    # the data is from Wengong's repo
    datapath = "iclr19-graph2graph/data/qed"
    if train_data_loader is None:
        train_data_loader = get_loader(datapath, "train", 48, True)
    if val_data_loader is None:
        val_data_loader = get_loader(datapath, "val", 48, True)

    # create your optimizer
    optimizer = torch.optim.Adam(molopt.parameters(), lr=0.01)

    for epoch in range(prev_epoch + 1, prev_epoch + args.n_epochs + 1):
        start = time.time()
        print ("Epoch:", epoch)

        # run the training procedure
        run_func(molopt, optimizer, train_data_loader, "train", args)

        # compute the validation loss as well, at the end of the epoch?
        run_func(molopt, optimizer, val_data_loader, "val", args)

        end = time.time()
        print("Epoch duration:", end - start)

        # save your progress along the way
        save_model(molopt, args, args.output_dir, "{}_{}".format(args.init_model, epoch))
    
    return molopt


def run_func(model, optim, data_loader, data_type, args):
    """ 
    Function that trains the GCN embeddings.
    Also used for validation purposes.
    """
    is_train = data_type == 'train'
    if is_train:
        model.train()
    else:
        model.eval()

    stats_tracker = StatsTracker()
    for _, i in enumerate(data_loader):
        if is_train:
            optim.zero_grad()   # zero the gradient buffers
        X = (MolGraph(i[0]))
        Y = (MolGraph(i[1]))

        loss = model.forward_train(X, Y)

        # add stat
        n_data = len(X.mols)
        stats_tracker.add_stat(data_type + '_mse', loss.item() * n_data, n_data)

        # in your training loop:
        if is_train:
            loss.backward()
            optim.step()    # Does the update
        
    stats_tracker.print_stats()


if __name__ == "__main__":
    main()