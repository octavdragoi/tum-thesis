import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-cuda', action='store_true', default=False,
                        help='Use gpu')
    parser.add_argument('-output_dir', default='', help='Output directory')
    parser.add_argument('-tb_logs_dir', default='', help='TensorBoard Logs directory')
    parser.add_argument('-init_model', type=str, default=None,
                    help='name of model to pick up from output directory; useful in training')
    parser.add_argument('-init_decoder_model', type=str, default=None,
                    help='name of decoder model to pick up from output directory; useful in training')
    parser.add_argument('-task', type = str, default = "qed",
                    help='name of dataset/task to run on',
                    choices=['logp04', 'logp06','qed','drd2'])
    parser.add_argument('-model_type', type = str, default = "ffn",
                    help='name of model to train',
                    choices=['ffn', 'transformer', 'slot'])

    # Prototype Params
    parser.add_argument('-pred_hidden', type=int, default=100,
                        help='Hidden dim for symbol prediction')
    parser.add_argument('-pc_hidden', type=int, default=50,
                        help='Hidden dim for point clouds, different from GCN hidden dim')
    parser.add_argument('-ffn_activation', type=str, choices=['ReLU', 'LeakyReLU'],
                        default='LeakyReLU')

    # General Model Params
    parser.add_argument('-n_epochs', type=int, default=10,
                        help='Number of epochs to train on')
    parser.add_argument('-dim_tangent_space', type=int, default=40,
                        help='Tangent space dimension for graph embeddings')


    # GCN Params
    parser.add_argument('-n_layers', type=int, default=5,
                        help='Number of layers in model')
    parser.add_argument('-n_hidden', type=int, default=50,
                        help='Size of hidden dimension for model')
    parser.add_argument('-n_ffn_hidden', type=int, default=100)
    parser.add_argument('-linear_out', action='store_true', default=False)
    parser.add_argument('-dropout_gcn', type=float, default=0.,
                        help='Amount of dropout for the model')
    parser.add_argument('-dropout_ffn', type=float, default=0.,
                        help='Dropout for final ffn layer')
    parser.add_argument('-agg_func', type=str, choices=['sum', 'mean', 'edge_sets'],
                        default='sum', help='aggregator function for atoms')
    parser.add_argument('-batch_norm', action='store_true', default=False,
                        help='Whether or not to normalize atom embeds')

    # Transformer Params
    parser.add_argument('-N_transformer', type = int, default = 6)
    parser.add_argument('-n_ffn_transformer', type = int, default = 100)
    parser.add_argument('-n_heads_transformer', type = int, default = 10)
    parser.add_argument('-dropout_transformer', type = float, default = 0.1)

    # OT Params
    parser.add_argument('-ot_solver', type=str, default='emd',
                        choices=['sinkhorn', 'sinkhorn_stabilized', 'emd',
                                 'greenkhorn', 'sinkhorn_epsilon_scaling'])
    parser.add_argument('-sinkhorn_entropy', type=float, default=1e-1,
                        help='Entropy regularization term for sinkhorn')
    parser.add_argument('-sinkhorn_max_it', type=int, default=10000,
                        help='Max num it for sinkhorn')

    # Penalty params
    parser.add_argument('-connectivity', type=bool, default=True)
    parser.add_argument('-valency', type=bool, default=True)
    parser.add_argument('-euler_characteristic_penalty', type=bool, default=True)
    parser.add_argument('-annealing_rate', type=float, default=0.05)
    parser.add_argument('-connectivity_lambda', type=float, default=0.025)
    parser.add_argument('-valency_lambda', type=float, default=0.07)
    parser.add_argument('-euler_lambda', type=float, default=0.3)
    parser.add_argument('-connectivity_hard', type=bool, default=False)
    parser.add_argument('-valency_hard', type=bool, default=False)

    args = parser.parse_args()
    args.device = 'cuda:0' if args.cuda else 'cpu'

    # what is this
    args.n_labels = 1

    return args