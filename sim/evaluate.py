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

    dinfo = avdata.DataInfo(repo)
    
    nmod = importlib.import_module('.networks.'+network, package='sim')
    model = nmod.model(dinfo)
        
    path = util.get_path(repo, network)
    fname = path+str(epoch)+'.h5'
    print("Loading network ("+fname+")")
    
    model.load_weights(fname)

    print("Evaluating negative only...")
    gen = avdata.SiameseGenerator(dinfo, evalset, inclPos=False)
    negsims = model.predict_generator(gen, verbose=1)
    negmean = np.mean(negsims[:,1])
    negvar  = np.var(negsims[:,1])
    print(" Mean = "+str(negmean))
    print(" Var  = "+str(negvar))

    print("Evaluating positive only...")
    gen = avdata.SiameseGenerator(dinfo, evalset, inclNeg=False)
    possims = model.predict_generator(gen, verbose=1)
    posmean = np.mean(possims[:,1])
    posvar  = np.var(possims[:,1])
    print(" Mean = "+str(posmean))
    print(" Var  = "+str(posvar))
    
    print("Evaluating all...")
    sims = np.vstack([negsims,possims])
    mean = np.mean(sims[:,1])
    var  = np.var(sims[:,1])
    print(" Mean = "+str(mean))
    print(" Var  = "+str(var))

    print("=> Mean similarity: "+str(mean))
