import argparse

def parse():
    parser = argparse.ArgumentParser(description='SFGL')
    parser.add_argument('-dataset', type=str, default='MDD', choices=['MDD', 'ABIDE'],
                        help='dataset')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--k_fold', type=int, default=5,
                        help='the fold number')
    parser.add_argument('--minibatch_size', type=int, default=4,
                        help='batch size')
    parser.add_argument('--window_size', type=int, default=30,
                        help='the length of the sliding window')
    parser.add_argument('--window_stride', type=int, default=2,
                        help='the stride of the sliding window')
    parser.add_argument('--dynamic_length', type=int, default=None,
                        help='available timeseries')
    parser.add_argument('--train_people', type=bool, default=True,
                        help='train personalized branch')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--reg_lambda', type=float, default=0.00001,
                        help='value of lambda')
    parser.add_argument('--gamma', type=float, default=0.8,
                        help='value of gamma')
    parser.add_argument('--clip_grad', type=float, default=0.0,
                        help='clip_grad')
    parser.add_argument('--num_epochs', type=int, default=5,
                        help='local epochs')
    parser.add_argument('--num_iters', type=int, default=10,
                        help='communication rounds')
    parser.add_argument('--num_heads', type=int, default=1,
                        help='the head number of Transformer')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='the number of GIN layer')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='output dimension of Transfomer')
    parser.add_argument('--ph', type=int, default=16,
                        help='hidden dimension of personalized branch')
    parser.add_argument('--sparsity', type=int, default=30,
                        help='degree of sparsity')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--readout', type=str, default='sero', choices=['garo', 'sero', 'mean'],
                        help='readout methods')
    parser.add_argument('--cls_token', type=str, default='sum', choices=['sum', 'mean', 'param'],
                        help='aggregation method of Transformer outputs')
    parser.add_argument('--Type', type=str, default='sum', choices=['sum', 'cat'],
                        help='aggregation method of personalized branch')
    parser.add_argument('--save_root_path', type=str, default=r'D:\zjhexp\result',
                        help='root directory of results')

    argv = parser.parse_args()
    return argv
