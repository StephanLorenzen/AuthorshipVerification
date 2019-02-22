import random
import numpy as np
import keras
from keras.preprocessing import sequence

from . import util

from sim.helpers.data import load_data

class WSGenerator(keras.utils.Sequence):
    def __init__(self, dinfo, filename, comp=1):
        self.datainfo = dinfo
        self.authors  = []

        self.comp = comp

        self.get_data(filename)
   
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

    def __len__(self):
        return len(self.authors)

    def __getitem__(self, index):
        uid, author = self.authors[index]
        
        ts = [x[0] for x in author]
        ls = [x[1] for x in author]
        X = self.__data_generation(author)
        Xs = cut(X, self.datainfo.batch_size())

        return (uid, ts, ls, Xs)

    def __data_generation(self, author):
        X = dict()
        head = [h[2] for h in author[:self.comp]] * len(author) # [t_1,t_2,...,t_comp] x n
        tail = [x[2] for x in author for i in range(self.comp)] # [t_1 x comp, t_2 x comp, ...]
        
        for cidx,c in enumerate(self.datainfo.channels()):
            h, t = self.prep_channel(cidx, head, tail)
            X['known_'+c+'_in'] = h
            X['unknown_'+c+'_in'] = t
        return X

    def prep_channel(self, cidx, head, tail):
        h, t = [h[cidx] for h in head], [x[cidx] for x in tail]
        
        h = sequence.pad_sequences(h, value=0, padding='post')
        t = sequence.pad_sequences(t, value=0, padding='post')
        
        return np.array(h), np.array(t)

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
