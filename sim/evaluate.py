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

def evaluate(args):
    config = util.get_config()

    network = args.NETWORK
    epoch = args.epoch
    repo = args.datarepo
    evalset = args.EVALSET
    inclneg, inclpos = True, True
    if args.negative:
        inclpos = False
    elif args.positive:
        inclneg = False

    dinfo = avdata.DataInfo(repo)
    
    nmod = importlib.import_module('.networks.'+network, package='sim')
    model = nmod.model(dinfo)
        
    path = util.get_path(repo, network)
    fname = path+str(epoch)+'.h5'
    print("Loading network ("+fname+")")
    
    model.load_weights(fname)

    print("Creating SIM-generator for "+str(evalset))
    gen = avdata.SiameseGenerator(dinfo, evalset, inclNeg=inclneg, inclPos=inclpos)
    print("Evaluating...")
    if args.negative:
        print(" Using only negative samples...")
    elif args.positive:
        print(" Using only positive samples...")
    sims = model.predict_generator(gen, verbose=1)
    mean = np.mean(sims[:,1])
    
    print("=> Mean similarity: "+str(mean))
