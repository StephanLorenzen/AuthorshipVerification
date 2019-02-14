import argparse
from .helpers import util
from .compute import compute as run_compute
from .cluster import cluster as run_cluster
from .predict import predict as run_predict

if __name__ == "__main__":
    config = util.get_config()

    parser = argparse.ArgumentParser(description='Clustering and prediction of writing style development.')
    
    subparser = parser.add_subparsers(dest='command')

    ### COMPUTE
    parser_comp = subparser.add_parser('compute')
    parser_comp.add_argument('NETWORK', type=str, help='Network to use.')
    parser_comp.add_argument('-e', '--epoch', metavar='EPOCH', type=str,
            default='final', help='Epoch network to use.')
    parser_comp.add_argument('-d', '--datarepo', metavar='DATAREPO', type=str,
            default=config['datarepo'], help='Data repository to use.')
    parser_comp.add_argument('-u', '--include_uid', action='store_const', const=True,
            default=False, help='If set, author id is included in output file')
    parser_comp.add_argument('DATASET', type=str, help='Data set.')

    ### CLUSTER
    parser_cluster = subparser.add_parser('cluster')
    parser_cluster.add_argument('NETWORK', type=str, help='Network to use.')
    parser_cluster.add_argument('-d', '--datarepo', metavar='DATAREPO', type=str,
            default=config['datarepo'], help='Data repository to use.')
    parser_cluster.add_argument('-dist', '--distance', metavar='DISTANCE', type=str,
            default='l1', choices=['l1','l2'], help='Distance function to use (default = l1).')
    parser_cluster.add_argument('-k', '--num_clusters', metavar='K', type=int,
            default=None, help='Number of clusters, runs select if none given.')
    parser_cluster.add_argument('DATASET', type=str, help='Data set.')
    
    ### PREDICT
    parser_predict = subparser.add_parser('predict')
    parser_predict.add_argument('NETWORK', type=str, help='Network to use.')
    parser_predict.add_argument('-e', '--epoch', metavar='EPOCH', type=str,
            default='final', help='Epoch network to use.')
    parser_predict.add_argument('-d', '--datarepo', metavar='DATAREPO', type=str,
            default=config['datarepo'], help='Data repository to use.')
    parser_predict.add_argument('DATASET', type=str, help='Data set.')

    args = parser.parse_args()

    if args.command == 'compute':
        run_compute(args)
    elif args.command == 'cluster':
        run_cluster(args)
    elif args.command == 'predict':
        run_predict(args) 
