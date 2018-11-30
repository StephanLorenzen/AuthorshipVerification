# External imports
import tensorflow as tf
from keras.callbacks import Callback
import numpy as np

import argparse
import importlib

# Local imports
from .helpers import data as avdata
from .helpers import util

def train(args):
    config = util.get_config()
    network = args.NETWORK
    datarepo = args.datarepo
    valset = args.VALSET
    trainset = args.TRAINSET
    subsample = args.subsample
    compstat = args.computestats
    restart = args.restart

    if compstat:
        print("Generating statistics")
        avdata.generate_stats(trainset, datarepo)
        print("Done!")

    
    nmod = importlib.import_module('.networks.'+network,package='sim')
    model = nmod.model(dinfo)

    print(model.summary())
    print("Training on "+str(trainset))
    print("Validating on "+str(valset))
    print("Batch size = "+str(dinfo.batch_size()))
    
    if restart > 0:
        print("Restarting from epoch "+str(restart))
        fname = util.get_path(datarepo, network)+str(restart)+'.h5'
        model.load_weights(fname)

    datagen = avdata.SiameseGenerator(dinfo, trainset, subsample=subsample)
    vdatagen = None
    if valset is not None:
        vdatagen = avdata.SiameseGenerator(dinfo, valset)
   
    model.fit_generator(generator=datagen, validation_data=vdatagen, epochs=40, verbose=1,
                            callbacks=[util.Checkpoint(datarepo, restart)])
