import numpy as np
import random
from .helpers import util, kmeans, quality

def cluster(args):
    network  = args.NETWORK
    repo     = args.datarepo
    dataset  = args.DATASET
    distance = args.distance
    k        = args.num_clusters
    nRemove  = max(args.num_remove, 0) # 0 <= nRemove  <= nCompare

    # Load data
    path = util.get_path(repo, dataset, network)
    data = []
    with open(path+'cluster-data.csv') as f:
        for l in f:
            l = l.strip().split(';')
            instance = []
            for elem in l[1:]:
                elem = elem.strip().split(',')
                instance.append([float(elem[0]),float(elem[1])])
            # Remove first nRemove elements 
            if nRemove >= len(instance):
                print("Warning: trying to remove too many elements. Skipping data point...")
                continue
            t0 = instance[nRemove][0]
            for i in range(len(instance)):
                instance[i][0]-=t0
            data.append(np.array(instance[nRemove:]))
    if k is None:
        kmeans.select(range(2,10), data, distance)
    else:
        err, labels, centers, centercnt, intervals = kmeans.cluster(data, k, distance, verbose=True)
    
        measures = quality.get_qualities(repo, dataset, k, labels, nRemove)

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
