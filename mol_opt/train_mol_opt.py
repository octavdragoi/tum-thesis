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

def format_name(model_name, epoch):
    return "model_{}_{}".format(model_name, epoch)

def get_latest_model(model_name, outdir):
    split_names = [x.split("_") for x in os.listdir(outdir)]
    split_names = [[x[0], "_".join(x[1:-1]), x[-1]] for x in split_names]
    try:
        max_epoch = max([int(x[2]) for x in split_names if x[0] == "model" and 
                            x[1] == model_name])
    except ValueError:
        return None, 0
    return format_name(model_name, max_epoch), max_epoch

def cleanup_dir(outdir, lastepoch):
    for fl in os.listdir(outdir):
        ep = int(fl.split("_")[2])
        ep_diff = lastepoch - ep
        for modulo in [10, 100]:
            if ep_diff > modulo and ep % modulo != 0:
                try:
                    os.remove(os.path.join(outdir, fl))
                except FileNotFoundError:
                    pass

def save_checkpoint(molopt, molopt_decoder, optimizer, penalty, scheduler, epoch, args, outfile):
    checkpoint = {
        'epoch': epoch,
        'args': args,
        'molopt': molopt.state_dict(),
        'molopt_decoder': molopt_decoder.state_dict(),
        'optimizer': optimizer.state_dict(),
        'penalty' : penalty.get_stats(),
        'scheduler': scheduler.state_dict() if scheduler is not None else None
    }
    torch.save(checkpoint, outfile)

def load_checkpoint(infile, args = None):
    checkpoint = torch.load(infile)
    args = checkpoint['args']
    molopt, molopt_decoder, optimizer, penalty, scheduler = initialize_models(args)
    molopt.load_state_dict(checkpoint['molopt'])
    molopt_decoder.load_state_dict(checkpoint['molopt_decoder'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler is not None and checkpoint['scheduler'] is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])
    else:
        scheduler = None
    penalty.load_stats(checkpoint['penalty'])
    return molopt, molopt_decoder, optimizer, penalty, scheduler, checkpoint['args'], checkpoint['epoch']

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
        molopt, molopt_decoder, optimizer, pen_loss, scheduler, _, prev_epoch = load_checkpoint(infile)
    else:
        print ("No model {} found in {}! Starting from scratch.".format(args.init_model, args.output_dir))
        molopt, molopt_decoder, optimizer, pen_loss, scheduler = initialize_models(args)


    # the data is from Wengong's repo
    datapath = "iclr19-graph2graph/data/qed"
    if train_data_loader is None:
        train_data_loader = get_loader(datapath, "train", args.batch_size, True)
    if val_data_loader is None or not args.one_batch_train:
        val_data_loader = get_loader(datapath, "valid", args.batch_size, True)

    tb_writer = SummaryWriter(logdir = args.tb_logs_dir)
    metrics = MolMetrics(SYMBOLS, FORMAL_CHARGES, BOND_TYPES, False)

    # write out hyperparameters
    if prev_epoch == 0:
        tb_writer.add_hparams(hparam_dict = vars(args), metric_dict = {})

    for epoch in range(prev_epoch + 1, prev_epoch + args.n_epochs + 1):
        start = time.time()
        print ("Epoch:", epoch)

        # run the training procedure
        scalars1 = run_func(molopt, molopt_decoder, optimizer, scheduler, train_data_loader, "train", 
                args, tb_writer, metrics, pen_loss, epoch)

        # compute the validation loss as well, at the end of the epoch?
        if not args.one_batch_train:
            scalars2 = run_func(molopt, molopt_decoder, optimizer, scheduler, val_data_loader, "val", args, 
                    tb_writer, metrics, pen_loss, epoch)
        else:
            scalars2 = {}
        
        if epoch == 1:
            scalars = {**scalars1, **scalars2}
            print (scalars)
            tb_writer.add_custom_scalars(scalars)
            print ("Added custom scalars metadata")

        end = time.time()
        print("Epoch duration:", end - start)

        # save your progress along the way
        outfile = os.path.join(args.output_dir, format_name(args.init_model, epoch))
        print (outfile)
        save_checkpoint(molopt, molopt_decoder, optimizer, pen_loss, scheduler, epoch, args, outfile)
        cleanup_dir(args.output_dir, epoch)
    
    return molopt, molopt_decoder


def run_func(mol_opt, mol_opt_decoder, optim, scheduler, data_loader, data_type, args, 
        tb_writer, metrics, pen_loss, epoch_idx):
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
    measure_stats_tracker = StatsTracker()
    metrics_stats_tracker = StatsTracker()
    mol_drawer = MolDrawer(tb_writer, SYMBOLS, BOND_TYPES, FORMAL_CHARGES)
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

        # add metric stats
        # we might want to do this only for the validation set
        measure_results = measure_task(X, pred_pack[0])
        for key in measure_results:
            measure_stats_tracker.add_stat(key, measure_results[key], n_data)
        # measure
        target = Y.get_graph_outputs()
        res, res_vars = metrics.measure_batch(pred_pack[0], target)
        for m in res:
            # quick and dirty hack for averaging
            n_data_curr = n_data if "degree" in m else 1
            metrics_stats_tracker.add_stat(m, res[m], n_data_curr)
        for m in res_vars:
            metrics_stats_tracker.add_var_stat(m, *res_vars[m])

        # in your training loop:
        if is_train:
            loss.backward()
            optim.step()    # Does the update
            if scheduler is not None:
                scheduler.step()

        if ((idx_batch == 0 and not is_train) or (idx_batch == 1000 and is_train))\
                and not args.one_batch_train: 

            # draw
            target_smiles = [Chem.MolToSmiles(y) for y in Y.rd_mols]
            if pairs:
                initial_smiles = [Chem.MolToSmiles(x) for x in X.rd_mols]
            else:
                initial_smiles = None
            mol_drawer.visualize_batch(pred_pack[0], target_smiles, epoch_idx, initial_smiles,
                text="{}-{}-".format(args.init_model, data_type))
        
        if idx_batch % 400 == 0 and not args.one_batch_train:
            losses_stats_tracker.print_stats("Losses idx_batch={}".format(idx_batch))
            measure_stats_tracker.print_stats("Measure idx_batch={}".format(idx_batch))
            metrics_stats_tracker.print_stats("Metrics idx_batch={}".format(idx_batch))
        
        # train on the first batch only
        if args.one_batch_train:
            break

    # log penalty statistics
    if is_train:
        pen_stats = pen_loss.get_stats()
        pen_loss.log()
        log_tensorboard(tb_writer, (pen_stats, {}), data_type + "_penalty", epoch_idx)

    losses_stats_tracker.print_stats("Losses Epoch {}, {}".format(epoch_idx, data_type))
    measure_stats_tracker.print_stats("Measure Epoch {}, {}".format(epoch_idx, data_type))
    metrics_stats_tracker.print_stats("Metrics Epoch {}, {}".format(epoch_idx, data_type))
    print ("Logits", [x.abs().mean().item() for x in yhat_logits])
    scalars1 = log_tensorboard(tb_writer, losses_stats_tracker.get_stats(), data_type + "_losses", epoch_idx)
    scalars1 = log_tensorboard(tb_writer, ({'learning_rate' : optim.param_groups[0]['lr']}, {}), data_type + "_penalty", epoch_idx)
    scalars2 = log_tensorboard(tb_writer, measure_stats_tracker.get_stats(), data_type + "_measure", epoch_idx)
    scalars3 = log_tensorboard(tb_writer, metrics_stats_tracker.get_stats(), data_type + "_metrics", epoch_idx)

    scalars = {data_type : {**scalars1, **scalars2, **scalars3}} if epoch_idx == 1 else None
    return scalars

if __name__ == "__main__":
    main()
