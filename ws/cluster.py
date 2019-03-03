import numpy as np
import random
from .helpers import util, kmeans, quality
from .helpers import data as cdata

def cluster(args):
    network  = args.NETWORK
    repo     = args.datarepo
    dataset  = args.DATASET
    distance = args.distance
    k        = args.num_clusters
    nCompare = max(1, args.num_compare)
    nRemove  = max(args.num_remove, 0)

    # Load data
    path = util.get_path(repo, dataset, network)
    uids, data = cdata.load_and_prep(path, nCompare, nRemove)
    if k is None:
        errs = kmeans.select(range(1,10), data, distance)
        perr = 1.0
        pdel = 1.0
        with open(path+'select.csv', 'w') as f:
            f.write('k;err;delta;imp\n')
            for kk, err in errs:
                delta = perr-err
                imp = delta/pdel
                perr = err
                pdel = delta
                f.write(str(kk)+';'+str(err)+';'+str(delta)+';'+str(imp)+'\n')
    else:
        err, labels, centers, centercnt, intervals = kmeans.cluster(data, k, distance, verbose=True)
        measures = quality.get_qualities(repo, dataset, k, labels, nRemove, uids)

        cnt = [0]*k
        cl  = max([x.shape[0] for x in centers])+1
        for l,x in zip(labels,data):
            cnt[l] += 1
        
        prefix = str(k)+'-'+str(distance)+"-"
        with open(path+prefix+'stats.txt', 'w') as f:
            f.write('Error:\n'+str(err)+'\n\n')
            f.write('Members:\n')
            for i in range(k):
                f.write(str(i)+':'+str(cnt[i])+'\n')
                clust = centers[i]
                upper,lower = intervals[i]
                with open(path+prefix+'c'+str(i)+'.csv', 'w') as cf:
                    cf.write('idx;sim;cnt;upper;lower;'+';'.join(list(measures.keys()))+'\n')
                    for j in range(clust.shape[0]):
                        cf.write(str(j*util.DELTA)+';'+str(clust[j])+';'
                                +str(float(centercnt[i][j])/cnt[i])+';'
                                +str(float(upper[j]))+';'+str(float(lower[j]))+';'
                                +';'.join([str(x[i][j]) for _, x in measures.items()])
                                +'\n')
