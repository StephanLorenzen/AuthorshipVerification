import numpy as np
import random
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
    return centers

def cluster(data, k, distname, maxiter=100, verbose=False):
    distfunc = wsdist.l1
   
    data = util._interpolate(data) 

    centers = random.sample(data, k)
    perr = 0.0
    for i in range(maxiter):
        if verbose:
            print("Iteration #"+str(i))
        labels, err = assign_clusters(data, centers, distfunc)
        centers = compute_centers(k, data, labels, distfunc)
        if verbose:
            print("=> Error = "+str(err))
        if abs(perr-err) < 0.000001:
            break
        perr = err
    return err, labels, centers

def select(ks, data, distfunc):
    errs = []
    print('Selecting k...')
    for k in ks:
        err, _, _ = kmeans(data, k, distfunc)
        errs.append(err)
        print('=> k = '+str(k)+', err = '+str(err))
    return list(zip(ks,errs))

