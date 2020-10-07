from pathlib import Path
import torch
import os
import time

from mol_opt.task_metrics import measure_task
from otgnn.utils import save_model, load_model, StatsTracker, log_tensorboard
from molgen.metrics.mol_metrics import MolMetrics
from tensorboardX import SummaryWriter
from otgnn.graph import SYMBOLS, FORMAL_CHARGES, BOND_TYPES

def format_name(model_name, epoch):
    return "model_{}_{}".format(model_name, epoch)

def format_data_name(model_name, epoch):
    return "data_{}_{}".format(model_name, epoch)

def get_latest_model(model_name, outdir, id = "model"):
    split_names = [x.split("_") for x in os.listdir(outdir)]
    split_names = [[x[0], "_".join(x[1:-1]), x[-1]] for x in split_names]
    # print (split_names)
    try:
        lst = [int(x[2]) for x in split_names if x[0] == id and 
                            x[1] == model_name]
        # print(lst)
        max_epoch = max([int(x[2]) for x in split_names if x[0] == id and 
                            x[1] == model_name])
    except ValueError:
        return None, 0
    return format_name(model_name, max_epoch), max_epoch

def cleanup_dir(outdir, lastepoch):
    for fl in os.listdir(outdir):
        ep = int(fl.split("_")[2])
        ep_diff = lastepoch - ep
        for modulo in [20, 100, 250]:
            if ep_diff > modulo and ep % modulo != 0:
                try:
                    os.remove(os.path.join(outdir, fl))
                except (FileNotFoundError, IsADirectoryError):
                    pass

def save_checkpoint(molopt, molopt_decoder, optimizer, penalty, recpen, crossatt, scheduler, epoch, args, outfile):
    checkpoint = {
        'epoch': epoch,
        'args': args,
        'molopt': molopt.state_dict(),
        'molopt_decoder': molopt_decoder.state_dict(),
        'optimizer': optimizer.state_dict(),
        'penalty' : penalty.get_stats(),
        'scheduler': scheduler.state_dict() if scheduler is not None else None
    }
    if recpen is not None:
        checkpoint['recpen'] = recpen.state_dict()
    if crossatt is not None:
        checkpoint['crossatt'] = crossatt.state_dict()
    torch.save(checkpoint, outfile)

def load_checkpoint(infile, init_fc, args = None, device = 'cuda:0'):
    checkpoint = torch.load(infile, map_location=torch.device(device))
    args = checkpoint['args']
    args.device = device
    molopt, molopt_decoder, optimizer, penalty, recpen, crossatt, scheduler = init_fc(args)
    molopt.load_state_dict(checkpoint['molopt'])
    molopt_decoder.load_state_dict(checkpoint['molopt_decoder'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler is not None and checkpoint['scheduler'] is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])
    else:
        scheduler = None
    penalty.load_stats(checkpoint['penalty'])
    if 'recpen' in checkpoint:
        recpen.load_state_dict(checkpoint['recpen'])
    if 'crossatt' in checkpoint:
        crossatt.load_state_dict(checkpoint['crossatt'])
    return molopt, molopt_decoder, optimizer, penalty, recpen, crossatt, scheduler, checkpoint['args'], checkpoint['epoch']

def save_data(path, X, pred_pack, losses, pen_stats, lr):
    # print (path, pred_pack, losses, pen_stats, lr)
    save_dict = {
        'X': X,
        'pred_pack': pred_pack,
        'losses': losses,
        'pen_stats': pen_stats,
        'lr' : lr
    }
    torch.save(save_dict, path)

def load_data(path, device = 'cuda:0'):
    load_dict = torch.load(path, map_location=torch.device(device))
    return load_dict

def do_epoch(epochidx, outdir, tb_writer, metrics, device = 'cuda:0'):
    datapoints = sorted([x for x in os.listdir(outdir) if "data" in x], key = lambda x: (len (x), x))
    for x in datapoints:
        curr_idx = int(x.split("_")[2])
        if curr_idx >= epochidx:
            print ("Epoch", curr_idx)
            epochidx = curr_idx
            
            # determine which files to log
            files = sorted(os.listdir(os.path.join(outdir, x)))
            to_log = []
            for f in files:
                if "complete" in f:
                    to_log.append(f.split("_")[0])
            for f in files:
                if "logged" in f:
                    to_log.remove(f.split("_")[0])
                    
            if epochidx == 1:
                scalars = {}
            for data_type in to_log:
                idx_batch = 0
                files = []
                measure_stats_tracker = StatsTracker()
                metrics_stats_tracker = StatsTracker()
                losses_stats_tracker = StatsTracker()
                while os.path.exists(os.path.join(outdir, x, "{}_{}.out".format(data_type, idx_batch))):
                    print ("{}_{}.out".format(data_type, idx_batch))
                    loaded_dict = load_data(os.path.join(outdir, x, "{}_{}.out".format(data_type, idx_batch)), device = device)
                    pred_pack = loaded_dict['pred_pack']
    #                 X = loaded_dict['X']
                    X = pred_pack[1] # TODO
                    Y = X
                    n_data = len(X.mols)
                
                    # get the losses
                    for key, val in loaded_dict['losses'][0].items():
                        # print (key, val)
                        losses_stats_tracker.add_stat(key, val, 1)
                        
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
                    idx_batch += 1

                # print and send logs to tensorboard
                losses_stats_tracker.print_stats("Losses Epoch {}, {}".format(epochidx, data_type))
                measure_stats_tracker.print_stats("Measure Epoch {}, {}".format(epochidx, data_type))
                metrics_stats_tracker.print_stats("Metrics Epoch {}, {}".format(epochidx, data_type))
            
                log_tensorboard(tb_writer, ({'learning_rate' : loaded_dict['lr']}, {}), data_type + "_penalty", epochidx)
                scalars1 = log_tensorboard(tb_writer, losses_stats_tracker.get_stats(), data_type + "_losses", epochidx)
                scalars2 = log_tensorboard(tb_writer, measure_stats_tracker.get_stats(), data_type + "_measure", epochidx)
                scalars3 = log_tensorboard(tb_writer, metrics_stats_tracker.get_stats(), data_type + "_metrics", epochidx)

                if data_type == "train":
                    log_tensorboard(tb_writer, (loaded_dict['pen_stats'], {}), data_type + "_penalty", epochidx)

                if epochidx == 1:
                    scalars[data_type] = {**scalars1, **scalars2, **scalars3}
                Path(os.path.join(outdir, x, "{}_logged.out".format(data_type))).touch()

        # if ((idx_batch == 0 and not is_train) or (idx_batch == 1000 and is_train))\
        #         and not args.one_batch_train: 

        #     # draw
        #     target_smiles = [Chem.MolToSmiles(y) for y in Y.rd_mols]
        #     if pairs:
        #         initial_smiles = [Chem.MolToSmiles(x) for x in X.rd_mols]
        #     else:
        #         initial_smiles = None
        #     mol_drawer.visualize_batch(pred_pack[0], target_smiles, epoch_idx, initial_smiles,
        #         text="{}-{}-".format(args.init_model, data_type))

    if epochidx == 1:
        print (scalars)
        tb_writer.add_custom_scalars(scalars)
        print ("Added custom scalars metadata")

    return epochidx

def main(outdir, logdir, prev_epoch = 1, flush_secs = 5, device = 'cuda:0'):
    tb_writer = SummaryWriter(logdir = logdir, flush_secs = flush_secs)
    metrics = MolMetrics(SYMBOLS, FORMAL_CHARGES, BOND_TYPES, False, device = device)
    curr_epoch = prev_epoch

    while 1:
        curr_epoch = do_epoch(curr_epoch, outdir, tb_writer, metrics, device = device)
        print ("Currently at epoch {}. Sleeping...".format(curr_epoch))
        time.sleep(60)

    tb_writer.close()
