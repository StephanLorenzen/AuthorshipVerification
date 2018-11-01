# External imports
import tensorflow as tf
from keras.callbacks import Callback
import numpy as np

import argparse

# Local imports
import helpers.data as avdata
from helpers.profiles import PROFILES
import helpers.util as util
from networks import n1,n2,n3,n4

def train(datafile, valdatafile=None, dataset="MaCom", network='n1'):
    profile = PROFILES[dataset]

    datagen = avdata.get_siamese_generator(datafile, dataset, channels=('char','word'), batch_size=16)
    
    if network == 'n1':
        model = n1.model(profile, datagen)
    elif network == 'n2':
        model = n2.model(profile, datagen)
    elif network == 'n3':
        model = n3.model(profile, datagen)
    elif network == 'n4':
        model = n4.model(profile, datagen)

    print(model.summary())

    vdatagen = None
    if valdatafile is not None:
        vdatagen = avdata.get_siamese_generator(valdatafile, dataset, channels=('char','word'),
                batch_size=16)
    
    model.fit_generator(generator=datagen, validation_data=vdatagen, epochs=40, verbose=1,
                            callbacks=[util.Checkpoint(dataset)])
    
def test(datafile, dataset="MaCom"):
    pass



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Catching Ghost Writers, YO!')
    parser.add_argument('-n', '--network', metavar='NETWORK', type=str,
        choices=['n1', 'n2', 'n3', 'n4'],
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
