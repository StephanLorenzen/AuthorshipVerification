import os
from configparser import ConfigParser
import numpy as np
import random

## Cluster methods
def assign_clusters(data, centers, distfunc):
    labels = []
    err    = 0.0
    for d in data:
        ds = [distfunc(d,c) for c in centers]
        ci = np.argmin(ds)
        labels.append(ci)
        err += ds[ci]
    return labels, err/len(data)

def compute_centers(k, data, labels, distfunc):
    mdim = max([d.shape[0] for d in data])
    centers = [np.zeros(mdim) for _ in range(k)]
    counts  = [np.zeros(mdim) for _ in range(k)]
    
    for d,l in zip(data,labels):
        centers[l][:d.shape[0]] += d
        counts[l][:d.shape[0]]  += np.ones(d.shape[0])

    for i in range(k):
        center = centers[i]
        count  = counts[i]
        for j in range(mdim):
            if count[j] == 0:
                break
        centers[i] = center[:j]/count[:j]
    return centers

def kmeans(data, k, distfunc, iterations=10, verbose=False):
    centers = random.sample(data, k)
    for i in range(iterations):
        if verbose:
            print("Iteration #"+str(i))
        labels, err = assign_clusters(data, centers, distfunc)
        centers = compute_centers(k, data, labels, distfunc)
        if verbose:
            print("=> Error = "+str(err))
    return err, labels, centers

## Distance functions
def l1(d1, d2):
    # d1, d2 np-arrays of similarities of varying length
    n = min(d1.shape[0], d2.shape[0])
    return np.sum(np.abs(d1[:n]-d2[:n]))

## General
def kselect(ks, data, distfunc):
    errs = []
    print('Selecting k...')
    for k in ks:
        err, _, _ = kmeans(data, k, distfunc)
        errs.append(err)
        print('=> k = '+str(k)+', err = '+str(err))
    return list(zip(ks,errs))

_CONFIG = None
def get_config():
    global _CONFIG
    if _CONFIG is not None:
        return _CONFIG
    _CONFIG = ConfigParser()
    _CONFIG.optionxform = str
    _CONFIG.read('config.ini')
    _CONFIG = _CONFIG['Config']
    return _CONFIG

def get_path(datarepo, dataset, network):
    config = get_config()
    path = config['path_storage']+'ws/'+datarepo+'/'+network+'-'+dataset+'/'
    if not os.path.exists(path):
        os.makedirs(path)
    return path

