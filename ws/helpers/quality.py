# Compute average quality graph for group of profiles
import numpy as np
from . import kmeans, util

def get_qualities(repo, dataset, k, labels, nRemove):
    meta = []
    rawpath = util.get_data_path(repo)
    path = util.get_path(repo, dataset, 'charCNN')

    with open(rawpath+dataset+'_wrdcnt.csv') as fwc, open(rawpath+dataset+'_meta.csv') as fmeta:
        fmeta.readline() # remove header
        for l in fmeta:
            meta.append(l.strip().split(';'))
        for i,l in enumerate(fwc):
            l = l.strip().split(';')
            meta[i].append(l[1])
            meta[i].append(l[2])

    authors = dict()
    for m in meta:
        assert(len(m) == 12)
        uid = m[0]
        nsen, nnoun, nverb = float(m[1]), float(m[2]), float(m[3])
        flesch  = float(m[4])
        smog    = float(m[5])
        coleman = float(m[6])
        ari     = float(m[7])
        linsear = float(m[8])
        gf      = float(m[9])
        wrdcnt  = (float(m[10])/float(m[11]))
        totwrd  = float(m[11])

        mesnoun = nnoun / nsen
        mesverb = nverb / nsen
        if uid not in authors:
            authors[uid] = []
        authors[uid].append([mesnoun, mesverb, flesch, smog, coleman, ari, linsear, gf, wrdcnt, totwrd, nsen])

    metas = []
    with open(path+'data-meta.csv') as f:
        for l in f:
            l = l.strip().split(';')[1:]
            xs = []
            for x in l:
                x = x.strip().split(',')
                xs.append((float(x[1]),float(x[0])))
            metas.append(xs)

    measures = dict()
    measures['sentences']=[]
    measures['words']    =[]
    measures['nouns']   = []
    measures['verbs']   = []
    measures['fleschs'] = []
    measures['smogs']   = []
    measures['colemans']= []
    measures['aris']    = []
    measures['linsears']= []
    measures['gfs']     = []
    measures['wrdcnts'] = []
    measures['mwl']     = []
    for (uid, ms), meta in zip(authors.items(), metas):
        ms = np.array(ms)
        ms = np.transpose(ms)
        ms = ms[:,nRemove:]
        meta = meta[nRemove:]
        t0 = meta[0][0]
        ts = [x[0]-t0 for x in meta]
        assert(ms.shape[1]==len(meta))
        mwls = [x[1]/w for w,x in zip(ms[9],meta)]
        
        measures['sentences'].append(list(zip(ts,ms[10])))
        measures['words'].append(list(zip(ts,ms[9])))
        measures['nouns'].append(list(zip(ts,ms[0])))
        measures['verbs'].append(list(zip(ts,ms[1])))
        measures['fleschs'].append(list(zip(ts,ms[2])))
        measures['smogs'].append(list(zip(ts,ms[3])))
        measures['colemans'].append(list(zip(ts,ms[4])))
        measures['aris'].append(list(zip(ts,ms[5])))
        measures['linsears'].append(list(zip(ts,ms[6])))
        measures['gfs'].append(list(zip(ts,ms[7])))
        measures['wrdcnts'].append(list(zip(ts,ms[8])))
        measures['mwl'].append(list(zip(ts, mwls)))

    for key in list(measures.keys()):
        measures[key] = util._interpolate(measures[key])

    for key in list(measures.keys()):
        measures[key], _ = kmeans.compute_centers(k, measures[key], labels)
    return measures
