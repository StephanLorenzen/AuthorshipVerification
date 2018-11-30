import random
import numpy as np
import configparser
import re
import datetime
import keras
from keras.preprocessing import sequence

from . import util

from sim.helpers.data import load_data

class AVGenerator(keras.utils.Sequence):
    def __init__(self, dinfo, filename):
        self.datainfo = dinfo
        self.authors  = []
        self.data     = []
        self.problems = []
        
        self.get_data(filename)
        self.construct_problems()
   
    def get_data(self, filename):
        self.authors = []
        auths = list(load_data(filename, self.datainfo.dataset,
            self.datainfo.channels(), incl_ts=True).items())
        
        for (uid, data) in auths:
            texts = []
            data.sort(key=lambda x: x[0])
            for d in data:
                ts = d[0]
                ls = len(d[1])
                proc = self.datainfo.encode(d[1:])
                texts.append((ts, ls, proc))
            self.authors.append((uid, texts))

    def construct_problems(self, prob=0.5):
        self.problems = []
        sprob = 1.0/(1.0-prob)-1.0
        for aidx, _ in enumerate(self.authors):
            self.problems.append((aidx,aidx,-1))
            if np.random.rand() < sprob:
                oidx = aidx
                while oidx == aidx:
                    oidx = np.random.randint(0,len(self.authors))
                self.problems.append((aidx,oidx,np.random.randint(0, len(self.authors[oidx]))))

    def __len__(self):
        return len(self.problems)

    def __getitem__(self, index):
        (a1idx,a2idx,tidx) = self.problems[index]
        label = 1 if a1idx == a2idx else 0
        uid     = self.authors[a1idx][0]
        knowns  = self.authors[a1idx][1][:-1]
        unknown = self.authors[a2idx][1][tidx]

        ts = [x[0] for x in knowns]
        ls = [x[1] for x in knowns]
        X = self.__data_generation(knowns, unknown)
        Xs = cut(X, self.datainfo.batch_size())

        return (uid, ts, ls, Xs, label)

    def __data_generation(self, knowns, unknown):
        X = dict()
        unknown = unknown[2]
        knowns  = [x[2] for x in knowns]
        for cidx,c in enumerate(self.datainfo.channels()):
            k, u = self.prep_channel(cidx, knowns, unknown)
            X['known_'+c+'_in'] = k
            X['unknown_'+c+'_in'] = u
        return X

    def prep_channel(self, cidx, knowns, unknown):
        k, u = [x[cidx] for x in knowns], [unknown[cidx]]*len(knowns)
        
        k = sequence.pad_sequences(k, value=0, padding='post')
        u = sequence.pad_sequences(u, value=0, padding='post')
        
        return np.array(k), np.array(u)

def load_similarities(repo, network, trainset):
    simfile = util.get_sim_path(repo)+network+'-'+trainset+'.csv'
    problems = []
    with open(simfile) as f:
        for l in f:
            l = l.strip().split(';')
            label = (l[1]=='1')
            preds = []
            for p in l[2:]:
                time,length,score = p.split(',')
                preds.append((int(time),int(length),float(score)))
            problems.append((label, preds))
    return problems

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

def info(datafile, datarepo):
    dinfo = DataInfo(datarepo)
    agen = AVGenerator(dinfo,datafile)
    print('#AV = '+str(len(agen)))
