# External imports
import tensorflow as tf
from keras.callbacks import Callback
import numpy as np

import argparse
import importlib
import os

# Local imports
import helpers.data as avdata
import helpers.util as util
from networks import n1,n2,n3,n4,n5

def get_weights(dinfo, network, epoch):
    nmod = importlib.import_module('networks.'+network)
    
    mod = nmod.model(dinfo)
   
    print(mod.summary())

    fname = 'store/'+dinfo.dataset+'/'+network+'-'+str(epoch)+'.h5'
    print("Loading weights ("+fname+")")
    mod.load_weights(fname)

    outdir = 'visualize/'+dinfo.dataset+'/'
    # create dir
    if not os.path.exists(outdir):
        os.makedirs(outdir) 
    
    for l in mod.layers:
        name = l.name
        ws = l.get_weights()
        if len(ws) == 0 or len(ws) >= 3:
            continue
        
        with open(outdir+network+'-'+name+".csv", 'w') as f:
            f.write('Weights\n')
            for w in ws[0]:
                f.write(';'.join([str(x) for x in w])+"\n")
            if len(ws) == 2:
                f.write('Bias\n')
                f.write(';'.join([str(x) for x in ws[1]])+"\n")



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Catching Ghost Writers, YO!')
    parser.add_argument('-d', '--datarepo', metavar='DATAREPO', type=str,
        choices=['PAN13', 'MaCom'], default='MaCom',
        help='Data repository to use (default MaCom).')
    parser.add_argument('NETWORK', type=str,
        help='Network to use.')
    parser.add_argument('-e', '--epoch', metavar='EPOCH', type=str, default='final',
            help='Epoch network to use.')
    parser.add_argument('MODE', type=str, choices=['get_weights', 'plot_training'],
        help='Visualize mode.')

    args = parser.parse_args()
    network = args.NETWORK
    epoch = args.epoch
    mode = args.MODE
    repo = args.datarepo
   
    dinfo = avdata.DataInfo(repo)
    
    if mode == 'get_weights':
        get_weights(dinfo, network, epoch)

