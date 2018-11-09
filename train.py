# External imports
import tensorflow as tf
from keras.callbacks import Callback
import numpy as np

import argparse
import importlib

# Local imports
import helpers.data as avdata
import helpers.util as util
from networks import n1,n2,n3,n4,n5

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Catching Ghost Writers, YO!')
    parser.add_argument('NETWORK', type=str, choices=['n1', 'n2', 'n3', 'n4', 'n5'],
        help='Network to use.')
    parser.add_argument('-d', '--datarepo', metavar='DATAREPO', type=str,
        choices=['PAN13', 'MaCom'], default='MaCom',
        help='Data repository to use.')
    parser.add_argument('TRAINSET', type=str, help='Training set file.')
    parser.add_argument('VALSET', type=str, default=None, help='Validation set file.')

    args = parser.parse_args()
    network = args.NETWORK
    datarepo = args.datarepo
    valset = args.VALSET
    trainset = args.TRAINSET

    dinfo = avdata.DataInfo(datarepo)
    
    nmod = importlib.import_module('networks.'+network)
    model = nmod.model(dinfo)

    print(model.summary())
    print("Training on "+str(trainset))
    print("Validating on "+str(valset))
    print("Batch size = "+str(dinfo.batch_size()))
    
    datagen = avdata.SiameseGenerator(dinfo, trainset)
    vdatagen = None
    if valset is not None:
        vdatagen = avdata.SiameseGenerator(dinfo, valset)
   
    model.fit_generator(generator=datagen, validation_data=vdatagen, epochs=40, verbose=1,
                            callbacks=[util.Checkpoint(datarepo)])
