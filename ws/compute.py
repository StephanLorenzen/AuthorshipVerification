import os
import importlib
import numpy as np

from .helpers import util
from .helpers.data import WSGenerator
from .helpers.util import DELTA

from sim.helpers.util import get_path as get_network_path
from sim.helpers.data import DataInfo

def compute(args):
    network = args.NETWORK
    epoch = args.epoch
    anon = not args.include_uid
    repo = args.datarepo
    dataset = args.DATASET

    dinfo = DataInfo(repo)
    wpath = get_network_path(repo, network)+str(epoch)+'.h5'
    nmod = importlib.import_module('sim.networks.'+network)
    model = nmod.model(dinfo)
    model.load_weights(wpath)

    print("Creating WS-generator for "+str(dataset))
    gen = WSGenerator(dinfo, dataset)

    res = []
    print("Generating similarities for "+str(dataset)+" with "+str(len(gen))+" authors.")
    per = 0
    for i, (uid, ts, ls, Xs) in enumerate(gen):
        if i >= per*len(gen):
            print(str(round(per*100))+"%")
            per += max(0.01, 1.0/len(gen))

        sims = np.empty((0,2))
        for x in Xs:
            sims = np.vstack([sims, model.predict(x)])

        # prediction done, fix times
        
        t0 = ts[0]
        tp = -1
        newts = []
        for t in ts:
            tc = (t-t0) / (60*60*24*30) # 1 month
            if tc-tp < DELTA:
                tc = tp+DELTA
            newts.append(tc)
            tp = tc
        ts = newts

        res.append((uid, ts, ls, sims))

    simOut  = util.get_path(repo, dataset, network)+'data-sim.csv'
    metaOut = util.get_path(repo, dataset, network)+'data-meta.csv'
    with open(simOut, 'w') as fsim, open(metaOut, 'w') as fmeta:
        for (uid,ts,ls,sims) in res:
            if anon:
                uid = 'author'
            fsim.write(str(uid)+';'+';'.join([str(sim[1]) for sim in sims])+'\n')
            fmeta.write(str(uid)+';'+';'.join([str(l)+','+str(t) for l,t in zip(ls,ts)])+'\n')

