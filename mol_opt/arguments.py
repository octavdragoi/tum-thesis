import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-cuda', action='store_true', default=False,
                        help='Use gpu')

    # Prototype Params
    parser.add_argument('-pc_hidden', type=int, default=50,
                        help='Hidden dim for point clouds, different from GCN hidden dim')
    parser.add_argument('-ffn_activation', type=str, choices=['ReLU', 'LeakyReLU'],
                        default='LeakyReLU')

    # General Model Params
    parser.add_argument('-n_epochs', type=int, default=10,
                        help='Number of epochs to train on')

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

    args = parser.parse_args()
    args.device = 'cuda:0' if args.cuda else 'cpu'

    # what is this
    args.n_labels = 1

    return args