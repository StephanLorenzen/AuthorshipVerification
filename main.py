# External imports
import tensorflow as tf
from keras.callbacks import Callback
import numpy as np

import argparse

# Local imports
import helpers.data as avdata
import helpers.util as util
from networks import n1,n2,n3,n4,n5

def train(datafile, valdatafile=None, dataset="MaCom", network='n1'):
    dinfo = avdata.DataInfo(dataset)
    
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

    print(model.summary())
    
    datagen = avdata.DataGenerator(dinfo, datafile)
    vdatagen = None
    if valdatafile is not None:
        vdatagen = avdata.DataGenerator(dinfo, valdatafile)
   
    model.fit_generator(generator=datagen, validation_data=vdatagen, epochs=40, verbose=1,
                            callbacks=[util.Checkpoint(dataset)])
    
def test(datafile, dataset="MaCom"):
    pass

def get_weights(network, iteration, dataset='MaCom'):
    dinfo = avdata.DataInfo(dataset)
    
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
   
    model.load_weights("store/"+dataset+"/n2-2.h5")
    import pdb; pdb.set_trace()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Catching Ghost Writers, YO!')
    parser.add_argument('-n', '--network', metavar='NETWORK', type=str,
        choices=['n1', 'n2', 'n3', 'n4', 'n5'],
        help='Network to use.')
    parser.add_argument('-d', '--data', metavar='DATA', type=str,
        choices=['PAN13', 'MaCom'], default='MaCom',
        help='Data to use.')
    parser.add_argument('-v', '--val', metavar='VALFILE', type=str, default=None,
        help='Validation set file.')
    parser.add_argument('training', type=str,
        help='Training set file.')

    args = parser.parse_args()
    network = args.network
    dataset = args.data
    valset = args.val
    trainset = args.training

    train(trainset, valset, dataset, network)
