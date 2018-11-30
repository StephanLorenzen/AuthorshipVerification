# External imports
import tensorflow as tf
from keras.callbacks import Callback
import numpy as np
import os

import argparse
import importlib

# Local imports
from .helpers import data as avdata
from .helpers import util

def test(args):
    config = util.get_config()

    network = args.NETWORK
    epoch = args.epoch
    repo = args.datarepo
    testset = args.TESTSET

    dinfo = avdata.DataInfo(repo)
    
    nmod = importlib.import_module('.networks.'+network, package='sim')
    model = nmod.model(dinfo)
        
    path = util.get_path(repo, network)
    fname = path+str(epoch)+'.h5'
    print("Loading network ("+fname+")")
    
    model.load_weights(fname)

    print("Creating SIM-generator for "+str(testset))
    gen = avdata.SiameseGenerator(dinfo, testset)
    print("Evaluating...")
    loss, accuracy = model.evaluate_generator(generator=gen)
    print("=> Loss: "+str(loss))
    print("=> Accuracy: "+str(accuracy))
   
    path = util.get_path(repo, network)
    fname = path+'results-'+str(epoch)+'-'+testset+'.csv'
    with open(fname, 'w') as out:
        out.write('Loss;Accuracy\n')
        out.write(str(loss)+';'+str(accuracy)+'\n')
