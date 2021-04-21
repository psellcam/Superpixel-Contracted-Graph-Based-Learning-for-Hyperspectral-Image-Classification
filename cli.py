import argparse

def create_parser():
    parser = argparse.ArgumentParser(description='Command Line Arguments')
    parser.add_argument('--dataset', default='Indiana', type=str,
                        choices=['Indiana', 'PaviaU' , 'Salinas'],
                        help='dataset name')
    parser.add_argument('--pca-v', '--pca-variance', default=0.998, type=float,
                        help='explained variance required via pca')
    parser.add_argument('--cm-feats', default=20, type=int,
                        help='number of features for covariance matrices')
    parser.add_argument('--cm-size', default=25, type=int,
                        help='size of window for covariance matrices')
    parser.add_argument('--cm-nn', default=400, type=int,
                        help='number of nearest neighbours for covariance matrices')
    parser.add_argument('--run-n', default=1, type=int,
                        help='number of repeats of the code for statistical analysis')
    parser.add_argument('--num-labeled', default=15, type=int,
                        help='number of ground truth labels')
    return parser

def parse_commandline_args():
    return create_parser().parse_args()

