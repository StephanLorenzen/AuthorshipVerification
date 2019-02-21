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
    mode = args.mode

    dinfo = DataInfo(repo)
    wpath = get_network_path(repo, network)+str(epoch)+'.h5'
    nmod = importlib.import_module('sim.networks.'+network)
    model = nmod.model(dinfo)
    model.load_weights(wpath)

    print("Creating WS-generator for "+str(dataset))
    gen = WSGenerator(dinfo, dataset, mode=mode)

    res = []
    print("Generating similarities for "+str(dataset)+" with "+str(len(gen))+" authors.")
    per = 0
    for i, (uid, ts, ls, Xs) in enumerate(gen):
        if i >= per*len(gen):
            print(str(round(per*100))+"%")
            per += max(0.01, 1.0/len(gen))

        sims = np.empty((0,2))
        for x in Xs:
            #import pdb; pdb.set_trace()
            sims = np.vstack([sims, model.predict(x)])

        if mode == 'triple':
            newsims = []
            for i in range(len(ts)):
                newsims.append(sum(sims[i:i+3])/3)
        sims = np.array(newsims)

        # First element is same - TODO check why this is not the case
        sims[0,0] = 0.0
        sims[0,1] = 1.0

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

    outfile = util.get_path(repo, dataset, network)+'cluster-data.csv'
    with open(outfile, 'w') as out:
        for (uid,ts,_,sims) in res:
            if anon:
                out.write('author;'+';'.join([str(t)+','+str(sim[1]) for t,sim in zip(ts,sims)])+'\n')
            else:
                out.write(str(uid)+';'+';'.join([str(t)+','+str(sim[1]) for t,sim in zip(ts,sims)])+'\n')

