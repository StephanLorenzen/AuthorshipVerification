import argparse
from .helpers import util
from .combine import train as run_train
from .combine import test as run_test

if __name__ == "__main__":
    config = util.get_config()

    parser = argparse.ArgumentParser(prog='av',description='Catching Ghost Writers, YO!')
    
    subparser = parser.add_subparsers(dest='command')

    ### TRAIN
    parser_train = subparser.add_parser('train')
    parser_train.add_argument('NETWORK', type=str, help='Network to use.')
    parser_train.add_argument('-e', '--epoch', metavar='EPOCH', type=str,
            default='final', help='Epoch network to use.')
    parser_train.add_argument('-d', '--datarepo', metavar='DATAREPO', type=str,
        default=config['datarepo'], help='Data repository to use.')
    parser_train.add_argument('-rc', '--recompute', action='store_const', const=True, 
        default=False, help='Force recomputation of similarities using sim module.')
    parser_train.add_argument('TRAINSET', type=str, help='Training set.')

    ### TEST
    parser_test = subparser.add_parser('test')
    parser_test.add_argument('NETWORK', type=str, help='Network to use.')
    parser_test.add_argument('-e', '--epoch', metavar='EPOCH', type=str,
            default='final', help='Epoch network to use.')
    parser_test.add_argument('-d', '--datarepo', metavar='DATAREPO', type=str,
            default=config['datarepo'], help='Data repository to use.')
    parser_test.add_argument('-rc', '--recompute', action='store_const', const=True,
        default=False, help='Force recomputation of similarities using sim module.')
    parser_test.add_argument('-l', '--lamb', metavar='LAMBDA', type=float,
            default=1.0, help='Lambda for exp combine strategy.')
    parser_test.add_argument('-dt', '--delta', type=float, default=None,
            help='Threshold. If not set, ROC curve is computed.')
    parser_test.add_argument('COMBINE', choices=['max','min','uniform','exp'], type=str,
            default=None, help='Combine strategy.')
    parser_test.add_argument('TESTSET', type=str, help='Test set.')

    args = parser.parse_args()

    if args.command == 'train':
        run_train(args)
    elif args.command == 'test':
        run_test(args)
