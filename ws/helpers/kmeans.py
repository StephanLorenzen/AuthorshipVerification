import numpy as np
import random
import math
from . import util, dist as wsdist

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
    return centers, counts

def get_all_distribution_intervals(conf, k, data, labels):
    intervals = []
    for j in range(k):
        intervals.append(get_distribution_interval(conf, [d for l,d in zip(labels,data) if l == j]))
    return intervals

def get_distribution_interval(conf, data):
    mdim = max([d.shape[0] for d in data])
    dists  = [[] for _ in range(mdim)]
    upper  = np.zeros(mdim)
    lower  = np.zeros(mdim)

    for d in data:
        dim = d.shape[0]
        for i in range(dim):
            dists[i].append(d[i])

    conf = conf/2
    for i in range(mdim):
        dists[i].sort()
        cnt = len(dists[i])
        dists[i] = dists[i][math.floor(cnt*conf):math.ceil(cnt*(1.0-conf))]
        upper[i] = dists[i][-1]
        lower[i] = dists[i][0]
    
    return upper, lower

def cluster(data, k, distname, maxiter=100, verbose=False):
    distfunc = wsdist.l1
    if distname == 'l2':
        distfunc = wsdist.l2

    data = util._interpolate(data) 

    centers = random.sample(data, k)
    perr = 0.0
    for i in range(maxiter):
        if verbose:
            print("Iteration #"+str(i))
        labels, err = assign_clusters(data, centers, distfunc)
        centers, counts = compute_centers(k, data, labels, distfunc)
        if verbose:
            print("=> Error = "+str(err))
        if abs(perr-err) < 0.000001:
            break
        perr = err

    intervals = get_all_distribution_intervals(0.1, k, data, labels)

    return err, labels, centers, counts, intervals

def select(ks, data, distfunc):
    errs = []
    print('Selecting k...')
    for k in ks:
        err, _, _, _, _ = cluster(data, k, distfunc)
        errs.append(err)
        print('=> k = '+str(k)+', err = '+str(err))
    return list(zip(ks,errs))

