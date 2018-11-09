# External imports
import tensorflow as tf
from keras.callbacks import Callback
import numpy as np
import os

import argparse
import importlib

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
    
    nmod = importlib.import_module('networks.'+network)
    model = nmod.model(dinfo)

    fname = 'store/'+repo+'/'+network+'-'+str(epoch)+'.h5'
    print("Loading network ("+fname+")")
    
    model.load_weights(fname)

    print("Creating generator for "+str(testset))
    gen = avdata.AVGenerator(dinfo, testset)
    res = []
    per = 0
    for i, (ts, X, label) in enumerate(gen):
        if i >= per*len(gen):
            print(str(round(per*100))+'%')
            per += max(0.01, 1.0/len(gen))
        ys = model.predict(X)
        res.append((ts,ys,label))

    path = 'predsys/'+repo+'/'
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path+network+'-'+testset+'.csv', 'w') as out:
        for (ts,ys,label) in res:
            out.write(str(label)+';'+';'.join([(str(t)+','+str(y[1])) for t,y in zip(ts,ys)])+'\n')
            

