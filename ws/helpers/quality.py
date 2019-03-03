# Compute average quality graph for group of profiles
import numpy as np
from . import kmeans, util
from .util import DELTA
import datetime

def get_qualities(repo, dataset, k, labels, nRemove, uids):
    meta = []
    rawpath = util.get_data_path(repo)
    path = util.get_path(repo, dataset, 'charCNN')

    with open(rawpath+dataset+'_ts.csv') as ft, open(rawpath+dataset+'_wrdcnt.csv') as fwc, open(rawpath+dataset+'_meta.csv') as fmeta:
        for l in ft:
            l = l.strip().split(';')
            uid = l[0]
            ts  = int(datetime.datetime.strptime(l[1], '%d-%m-%Y').timestamp())
            meta.append([uid, ts])
        fmeta.readline() # remove header
        for i,l in enumerate(fmeta):
            l = l.strip().split(';')
            assert(meta[i][0] == l[0])
            meta[i] += [float(x) for x in l[1:]]
        for i,l in enumerate(fwc):
            l = l.strip().split(';')
            assert(meta[i][0] == l[0])
            meta[i] += [float(l[1]), float(l[2]), float(l[3])]

    authors = dict()
    for m in meta:
        assert(len(m) == 14)
        uid = m[0]
        ts  = m[1]
        nsen, nnoun, nverb = float(m[2]), float(m[3]), float(m[4])
        flesch  = float(m[5])
        smog    = float(m[6])
        coleman = float(m[7])
        ari     = float(m[8])
        linsear = float(m[9])
        gf      = float(m[10])
        wrdcnt  = (float(m[11])/float(m[12]))
        totwrd  = float(m[12])
        mwrdl   = float(m[13])/totwrd

        mesnoun = nnoun / nsen
        mesverb = nverb / nsen
        if uid not in authors:
            authors[uid] = []
        authors[uid].append([ts, mesnoun, mesverb, flesch, smog, coleman, ari, linsear, gf, wrdcnt, mwrdl, totwrd, nsen])

    for uid, ls in authors.items():
        ls.sort(key=lambda x: x[0])
        t0 = ls[0][0]
        tp = -1
        for x in ls:
            x[0] -= t0
            x[0] /= (60*60*24*30)
            if x[0]-tp < DELTA:
                x[0] = tp+DELTA
            tp = x[0]

    #metas = []
    #with open(path+'data-meta.csv') as f:
    #    for l in f:
    #        l = l.strip().split(';')[1:]
    #        xs = []
    #        for x in l:
    #            x = x.strip().split(',')
    #            xs.append((float(x[1]),float(x[0])))
    #        metas.append(xs)

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
    for uid in uids:
        assert(uid in authors)
        ms = authors[uid]
        ms = np.array(ms)
        ms = np.transpose(ms)
        t0 = ms[0,nRemove]
        ms = ms[:,nRemove:]
        [ts, mesnoun, mesverb, flesch, smog, coleman, ari, linsear, gf, wrdcnt, mwrdl, totwrd, nsen] = ms
        ts = [x-t0 for x in ts]
        assert(len(ts)==len(mwrdl))
        
        measures['sentences'].append(list(zip(ts,nsen)))
        measures['words'].append(list(zip(ts,totwrd)))
        measures['nouns'].append(list(zip(ts,mesnoun)))
        measures['verbs'].append(list(zip(ts,mesverb)))
        measures['fleschs'].append(list(zip(ts,flesch)))
        measures['smogs'].append(list(zip(ts,smog)))
        measures['colemans'].append(list(zip(ts,coleman)))
        measures['aris'].append(list(zip(ts,ari)))
        measures['linsears'].append(list(zip(ts,linsear)))
        measures['gfs'].append(list(zip(ts,gf)))
        measures['wrdcnts'].append(list(zip(ts,wrdcnt)))
        measures['mwl'].append(list(zip(ts, mwrdl)))

    for key in list(measures.keys()):
        measures[key] = util._interpolate(measures[key])

    for key in list(measures.keys()):
        measures[key], _ = kmeans.compute_centers(k, measures[key], labels)
    return measures
