import argparse
from .helpers import util
from .train import train as run_train
from .test import test as run_test
from .evaluate import evaluate as run_eval

if __name__ == "__main__":
    config = util.get_config()

    parser = argparse.ArgumentParser(description='Computing writing style similarity using deep learning.')
    
    subparser = parser.add_subparsers(dest='command')

    ### TRAIN
    parser_train = subparser.add_parser('train')
    parser_train.add_argument('NETWORK', type=str, help='Network to use.')
    parser_train.add_argument('-d', '--datarepo', metavar='DATAREPO', type=str,
            default=config['datarepo'], help='Data repository')
    parser_train.add_argument('-sub', '--subsample', metavar='PROB', type=float,
            help='Fraction of training data to use in each epoch.')
    parser_train.add_argument('-stat', '--computestats', metavar='DATASET',
            default=None, help='Compute statistics on DATASET (default=TRAINSET)')
    parser_train.add_argument('-r', '--restart', metavar='EPOCH', type=int, default=0,
            help='Restart training from given epoch.')
    parser_train.add_argument('TRAINSET', type=str, help='Training set.')
    parser_train.add_argument('VALSET', type=str, help='Validation set.')

    ### TEST
    parser_test  = subparser.add_parser('test')
    parser_test.add_argument('NETWORK', type=str, help='Network to use.')
    parser_test.add_argument('-e', '--epoch', metavar='EPOCH', type=str,
            default='final', help='Epoch network to use.')
    parser_test.add_argument('-d', '--datarepo', metavar='DATAREPO', type=str,
            default=config['datarepo'], help='Data repository to use.')
    parser_test.add_argument('TESTSET', type=str, help='Test set.')

    ### EVAL
    parser_eval  = subparser.add_parser('eval')
    parser_eval.add_argument('NETWORK', type=str, help='Network to use.')
    parser_eval.add_argument('-e', '--epoch', metavar='EPOCH', type=str,
            default='final', help='Epoch network to use.')
    parser_eval.add_argument('-d', '--datarepo', metavar='DATAREPO', type=str,
            default=config['datarepo'], help='Data repository to use.')
    parser_eval.add_argument('EVALSET', type=str, help='Test set.')
    args = parser.parse_args()

    parser_preprocess = subparser.add_parser('preprocess')

    if args.command == 'train':
        run_train(args)
    elif args.command == 'test':
        run_test(args)
    elif args.command == 'eval':
        run_eval(args)
    else:
        pass
        #run_preprocess(args)
