import numpy as np

from .helpers import util

def cluster(args):
    network  = args.NETWORK
    repo     = args.datarepo
    dataset  = args.DATASET
    distance = args.distance
    k        = args.num_clusters

    distfunc = util.l1

    # Load data
    path = util.get_path(repo, dataset, network)
    data = []
    with open(path+'cluster-data.csv') as f:
        for l in f:
            l = l.strip().split(';')
            instance = []
            for elem in l:
                elem = elem.strip().split(',')
                #instance.append((int(elem[0]),float(elem[1])))
                instance.append(float(elem[1]))
            data.append(np.array(instance))
    
    if k is None:
        util.kselect(range(2,10), data, util.l1)
    else:
        err, labels, centers = util.kmeans(data, k, util.l1, verbose=True)
        cnt = [0]*k
        for l in labels:
            cnt[l] += 1

        prefix = str(k)+'-'+str(distance)+"-"
        with open(path+prefix+'stats.txt', 'w') as f:
            f.write('Error:\n'+str(err)+'\n\n')
            f.write('Members:\n')
            for i in range(k):
                f.write(str(i)+':'+str(cnt[i])+'\n')
                clust = centers[i]
                with open(path+prefix+'c'+str(i)+'.csv', 'w') as cf:
                    cf.write('idx;sim\n')
                    for j in range(clust.shape[0]):
                        cf.write(str(j)+';'+str(clust[j])+'\n')
