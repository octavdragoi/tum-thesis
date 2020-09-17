import os
from pathlib import Path
import sys
# need the path to otgnn repo
sys.path.append(os.path.join(os.getcwd(), "otgnn")) 
sys.path.append(os.path.join(os.getcwd(), "molgen")) 
sys.path.append(os.path.join(os.getcwd(), "RAdam")) 
sys.path.append(os.path.join(os.getcwd(), "RangerDeepLearningOptimizer")) 

from otgnn.models import GCN
from otgnn.graph import MolGraph
from otgnn.utils import save_model, load_model, StatsTracker, log_tensorboard
from otgnn.graph import SYMBOLS, FORMAL_CHARGES, BOND_TYPES

from mol_opt.data_mol_opt import MolOptDataset
from mol_opt.data_mol_opt import get_loader
from mol_opt.arguments import get_args
from mol_opt.mol_opt import MolOpt
from mol_opt.decoder_mol_opt import MolOptDecoder
from mol_opt.ot_utils import FGW 
from mol_opt.task_metrics import measure_task
from mol_opt.log_mol_opt import save_data, format_name, format_data_name, get_latest_model, cleanup_dir, save_checkpoint, load_checkpoint

from molgen.metrics.Penalty import Penalty
from molgen.metrics.mol_metrics import MolMetrics
from molgen.dataloading.mol_drawer import MolDrawer

from RAdam.radam import RAdam
from RangerDeepLearningOptimizer.ranger import Ranger
from RangerDeepLearningOptimizer.ranger import RangerVA

from rdkit import Chem

from tensorboardX import SummaryWriter

import torch
import time

ft = {
    "SYMBOLS" : SYMBOLS,
    "BOND_TYPES" : BOND_TYPES,
    "FORMAL_CHARGES" : FORMAL_CHARGES
}

def initialize_models(args):
    molopt = MolOpt(args).to(device = args.device)
    molopt_decoder = MolOptDecoder(args).to(device = args.device)
    molopt_module_list = torch.nn.ModuleList([molopt, molopt_decoder])
    # create your optimizer
    # optimizer = torch.optiG.RMSprop(molopt_module_list.parameters(), lr=0.004)
    optimizer = torch.optim.Adam(molopt_module_list.parameters(), lr=0.004,
        amsgrad = False, weight_decay=0)
    # optimizer = torch.optim.AdamW(molopt_module_list.parameters(), lr=0.004,
    #     amsgrad = False, weight_decay= 0)
    # optimizer = torch.optim.SGD(molopt_module_list.parameters(), lr=0.004,
    #     momentum = 0.1, dampening = 0.1, nesterov = True)
    # optimizer = RAdam(molopt_module_list.parameters(), lr=0.004)
    # optimizer = Ranger(molopt_module_list.parameters(), lr=0.001, N_sma_threshhold=4, use_gc = False)
    # optimizer = RangerVA(molopt_module_list.parameters(), lr=0.007, k=10,n_sma_threshhold=4)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 100, gamma = 0.98)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, np.arange(0,2000, 100), gamma = 0.9)
    # lmbda = lambda epoch: 1.0 if epoch < 1300 else 0.2
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lmbda)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 
    #     factor = 0.5)
    # scheduler = None
    penalty = Penalty(args)
    # penalty = None

    return molopt, molopt_decoder, optimizer, penalty, scheduler

def main(args = None, train_data_loader = None, val_data_loader = None):
    if args is None:
        args = get_args()
        args.output_dir = "mol_opt/output"
        args.tb_logs_dir = "mol_opt/logs"

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.tb_logs_dir).mkdir(parents=True, exist_ok=True)

    # if model is not created yet, then create a new one
    prev_epoch = -1 

    model_name, prev_epoch = get_latest_model(args.init_model, args.output_dir)
    if model_name is not None:
        infile = os.path.join(args.output_dir, format_name(args.init_model, prev_epoch))
        print ("Found previous model {}, epoch {}. Overwriting args.".format(infile, prev_epoch))
        molopt, molopt_decoder, optimizer, pen_loss, scheduler, _, prev_epoch = load_checkpoint(infile, initialize_models)
    else:
        print ("No model {} found in {}! Starting from scratch.".format(args.init_model, args.output_dir))
        molopt, molopt_decoder, optimizer, pen_loss, scheduler = initialize_models(args)


    # the data is from Wengong's repo
    datapath = "iclr19-graph2graph/data/qed"
    if train_data_loader is None:
        train_data_loader = get_loader(datapath, "train", args.batch_size, True)

    for epoch in range(prev_epoch + 1, prev_epoch + args.n_epochs + 1):
        start = time.time()
        print ("Epoch:", epoch)

        Path(os.path.join(args.output_dir, format_data_name(args.init_model, epoch))).mkdir(parents=True, exist_ok=True)
        # run the training procedure
        run_func(molopt, molopt_decoder, optimizer, scheduler, train_data_loader, "train", 
                args, pen_loss, epoch)

        # compute the validation loss as well, at the end of the epoch?
        if not args.one_batch_train and val_data_loader is not None:
            run_func(molopt, molopt_decoder, optimizer, scheduler, val_data_loader, "val", args, 
                    pen_loss, epoch)

        end = time.time()
        print("Epoch duration:", end - start)

        # save your progress along the way
        outfile = os.path.join(args.output_dir, format_name(args.init_model, epoch))
        print ("Saving model, do not interrupt...")
        save_checkpoint(molopt, molopt_decoder, optimizer, pen_loss, scheduler, epoch, args, outfile)
        print ("Saved at", outfile)
        cleanup_dir(args.output_dir, epoch)
    
    return molopt, molopt_decoder


def run_func(mol_opt, mol_opt_decoder, optim, scheduler, data_loader, data_type, args, 
        pen_loss, epoch_idx):
    """ 
    Function that trains the GCN embeddings.
    Also used for validation purposes.
    """
    is_train = data_type == 'train'
    if is_train:
        # pairs = True
        pairs = False
        mol_opt.train()
        mol_opt_decoder.train()
    else:
        pairs = False
        mol_opt.eval()
        mol_opt_decoder.eval()

    fgw_loss = FGW(alpha = 0.5)

    losses_stats_tracker = StatsTracker()
    for idx_batch, i in enumerate(data_loader):
        if is_train:
            optim.zero_grad()   # zero the gradient buffers

        if pairs:
            X = (MolGraph(i[0]))
            Y = (MolGraph(i[1]))
        else:
            X = MolGraph(i)
            Y = X
        n_data = len(X.mols)

        x_embedding = mol_opt.forward(X)
        yhat_logits = mol_opt_decoder.forward(x_embedding, X, Y)
        if args.penalty_gumbel:
            yhat_labels = mol_opt_decoder.discretize_gumbel(*yhat_logits, tau = pen_loss.tau)
        else:
            yhat_labels = mol_opt_decoder.discretize_argmax(*yhat_logits)
        pred_pack = (yhat_labels, yhat_logits, Y.scope), Y
        # model_loss = fgw_loss(*pred_pack, tau = pen_loss.tau)
        model_loss = fgw_loss(*pred_pack, tau = 1)
        # compute the lambdas and losses, based on this fgw loss
        con_loss, val_loss, eul_loss = pen_loss(*pred_pack, epochidx = epoch_idx)
        # model_loss = fgw_loss(*pred_pack)

        loss = model_loss + pen_loss.conn_lambda * con_loss + \
            pen_loss.valency_lambda * val_loss + pen_loss.euler_lambda * eul_loss
        pen_loss.compute_lambdas(epoch_idx, model_loss.item()/n_data, 
            con_loss.item()/n_data, val_loss.item()/n_data, eul_loss.item()/n_data,
            loss.item()/n_data)

        # add stat
        losses_stats_tracker.add_stat('fgw', model_loss.item(), n_data)
        losses_stats_tracker.add_stat('conn_penalty', con_loss.item(), n_data)
        losses_stats_tracker.add_stat('val_penalty', val_loss.item(), n_data)
        losses_stats_tracker.add_stat('euler_penalty', eul_loss.item(), n_data)
        losses_stats_tracker.add_stat('total', loss.item(), n_data)

        # in your training loop:
        if is_train:
            loss.backward()
            optim.step()    # Does the update
            if scheduler is not None:
                scheduler.step()

        # output the metrics to a file
        data_path = os.path.join(args.output_dir, format_data_name(args.init_model, epoch_idx), "{}_{}.out".format(data_type, idx_batch))
        save_data(data_path, X, pred_pack, losses_stats_tracker.get_stats(), pen_loss.get_stats(), optim.param_groups[0]['lr'])

        # train on the first batch only
        if args.one_batch_train:
            break

    # signal that the output files are complete
    Path(os.path.join(args.output_dir, format_data_name(args.init_model, epoch_idx), "{}_{}.out".format(data_type, "complete"))).touch()

    # log penalty statistics
    if is_train:
        pen_loss.log()

    losses_stats_tracker.print_stats("Losses Epoch {}, {}".format(epoch_idx, data_type))

if __name__ == "__main__":
    main()
