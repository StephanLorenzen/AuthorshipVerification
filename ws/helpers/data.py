import random
import numpy as np
import keras
import math
from keras.preprocessing import sequence

from . import util
from .util import DELTA

from sim.helpers.data import load_data

class WSGenerator(keras.utils.Sequence):
    def __init__(self, dinfo, filename):
        self.datainfo = dinfo
        self.authors  = []

        self.get_data(filename)
   
    def get_data(self, filename):
        self.authors = []
        auths = list(load_data(filename, self.datainfo.dataset,
            self.datainfo.channels(), incl_ts=True).items())
        
        for (uid, data) in auths:
            texts = []
            data.sort(key=lambda x: x[0])
            t0 = data[0][0]
            tp = -1
            for d in data:
                ts = (d[0]-t0) / (60*60*24*30)
                if ts-tp < DELTA:
                    ts = tp+DELTA
                tp = ts
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
        txt1s = []
        txt2s = []
        for i in range(len(author)):
            for j in range(i+1,len(author)):
                txt1s.append(author[i][2])
                txt2s.append(author[j][2])

        for cidx,c in enumerate(self.datainfo.channels()):
            t1, t2 = self.prep_channel(cidx, txt1s, txt2s)
            X['known_'+c+'_in'] = t1
            X['unknown_'+c+'_in'] = t2
        return X

    def prep_channel(self, cidx, head, tail):
        h, t = [h[cidx] for h in head], [x[cidx] for x in tail]
        
        h = sequence.pad_sequences(h, value=0, padding='post')
        t = sequence.pad_sequences(t, value=0, padding='post')
        
        return np.array(h), np.array(t)

class WSRandGenerator(keras.utils.Sequence):
    def __init__(self, dinfo, filename, numSamples):
        self.datainfo = dinfo
        self.authors  = []
        self.numSamples = numSamples

        self.get_data(filename)

    def get_data(self, filename):
        self.authors = []
        auths = list(load_data(filename, self.datainfo.dataset,
            self.datainfo.channels(), incl_ts=True).items())
        
        for (uid, data) in auths:
            texts = []
            data.sort(key=lambda x: x[0])
            t0 = data[0][0]
            tp = -1
            for d in data:
                ts = (d[0]-t0) / (60*60*24*30)
                if ts-tp < DELTA:
                    ts = tp+DELTA
                tp = ts
                ls = len(d[1])
                proc = self.datainfo.encode(d[1:])
                texts.append((ts, ls, proc))
            self.authors.append((uid, texts))

    def __len__(self):
        return math.ceil(self.numSamples/self.datainfo.batch_size())

    def __getitem__(self, index):
        ts, X = self.__data_generation(self.datainfo.batch_size())
        return (ts, X)

    def __data_generation(self, size):
        X = dict()
        ts    = []
        txt1s = []
        txt2s = []
        for _ in range(size):
            a1 = random.randint(0,len(self.authors)-1)
            a2 = a1
            while a2 == a1:
                a2 = random.randint(0,len(self.authors)-1)
            (ts1,_,txt1) = random.choice(self.authors[a1][1])
            (ts2,_,txt2) = random.choice(self.authors[a2][1])
            ts.append((ts1,ts2))
            txt1s.append(txt1)
            txt2s.append(txt2)

        for cidx,c in enumerate(self.datainfo.channels()):
            t1, t2 = self.prep_channel(cidx, txt1s, txt2s)
            X['known_'+c+'_in'] = t1
            X['unknown_'+c+'_in'] = t2
        return ts, X

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

def load_and_prep(path, nCompare=1, nRemove=0):
    profiles = []
    with open(path+'data-meta.csv') as f:
        for l in f:
            profile = []
            l = l.split(';')
            uid = l[0]
            l = l[1:]
            for d in l:
                t = float(d.split(',')[1])
                profile.append(t)
            profiles.append((uid,profile))
    
    data = []
    with open(path+'data-sim.csv') as f:
        for i,l in enumerate(f):
            uid,profile = profiles[i]
            n = len(profile)
            l = l.split(';')[1:]
            cl = 0
            mp = dict()
            for i1 in range(0,n):
                mp[(i1,i1)] = 1.0
                for i2 in range(i1+1,n):
                    mp[(i1,i2)] = float(l[cl])
                    mp[(i2,i1)] = float(l[cl])
                    cl += 1
            for i1 in range(0,n):
                # i1 current data point
                ts = profile[i1]
                sim = 0.0
                for i2 in range(0,nCompare):
                    sim += mp[(i1,i2)]
                sim /= nCompare
                profile[i1] = (ts,sim) 
            data.append((uid,profile))

    # Remove if needed
    uids = []
    for i in range(len(data)):
        uid, profile = data[i]
        t0 = profile[nRemove][0]
        profile = [(ts-t0, sim) for ts, sim in profile]
        data[i] = np.array(profile[nRemove:])
        uids.append(uid)
    return uids,data

