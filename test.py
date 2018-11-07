# External imports
import tensorflow as tf
from keras.callbacks import Callback
import numpy as np

import argparse

# Local imports
import helpers.data as avdata
import helpers.util as util
from networks import n1,n2,n3,n4,n5

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Testing how to catch Ghost Writers, YO!')
    parser.add_argument('NETWORK', type=str, choices=['n1', 'n2', 'n3', 'n4', 'n5'],
            help='Network to use.')
    parser.add_argument('-e', '--epoch', metavar='EPOCH', type=str, default='final',
            help='Epoch network to use.')
    parser.add_argument('-d', '--datarepo', metavar='DATAREPO', type=str,
        choices=['PAN13', 'MaCom'], default='MaCom',
        help='Data repository to use (default MaCom).')
    parser.add_argument('TESTSET', type=str, help='Test set file.')

    args = parser.parse_args()
    network = args.NETWORK
    epoch = args.epoch
    repo = args.datarepo
    testset = args.TESTSET

    dinfo = avdata.DataInfo(repo)
    
    if network == 'n1':
        model = n1.model(dinfo)
    elif network == 'n2':
        model = n2.model(dinfo)
    elif network == 'n3':
        model = n3.model(dinfo)
    elif network == 'n4':
        model = n4.model(dinfo)
    elif network == 'n5':
        model = n5.model(dinfo)
   
    fname = 'store/'+repo+'/'+network+'-'+str(epoch)+'.h5'
    print("Loading network ("+fname+")")
    
    model.load_weights(fname)


