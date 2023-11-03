import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='', help="name of dataset")
    parser.add_argument('--method', type=str, default='', help="method")
    parser.add_argument('--submethod', type=str, default='', help="pcs submethod")
    parser.add_argument('--log', type=str, default='logger', help="logger filename")
    args = parser.parse_args()
    return args
