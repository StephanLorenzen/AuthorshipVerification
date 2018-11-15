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

def cut(X, batch_size):
    keys = list(X.keys())
    l = len(X[keys[0]])
    if l <= batch_size:
        return [X]
    res = []
    i = 0
    while l > 0:
        sX = dict()
        for k in keys:
            sX[k] = X[k][i*batch_size:(i+1)*batch_size]
        res.append(sX)
        i += 1
        l -= batch_size
    return res


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Testing how to catch Ghost Writers, YO!')
    parser.add_argument('NETWORK', type=str,
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
    for i, (uid, ts, X, label) in enumerate(gen):
        if i >= per*len(gen):
            print(str(round(per*100))+'%')
            per += max(0.01, 1.0/len(gen))
        Xs = cut(X, dinfo.batch_size())
        ys = np.empty((0,2))
        for x in Xs:
            ys = np.vstack([ys, model.predict(x)])
        res.append((uid,ts,ys,label))

    path = 'predsys/'+repo+'/'
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path+network+'-'+testset+'.csv', 'w') as out:
        for (uid,ts,ys,label) in res:
            out.write(str(uid)+";"+str(label)+';'+';'.join([(str(t)+','+str(y[1])) for t,y in zip(ts,ys)])+'\n')
            

