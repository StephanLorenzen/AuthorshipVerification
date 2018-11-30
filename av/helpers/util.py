import os
import importlib
import numpy as np
from configparser import ConfigParser

from . import data as avdata

from sim.helpers.util import get_path as get_network_path
from sim.helpers.data import DataInfo

def compute_similarities(repo, testset, network, epoch):
    dinfo = DataInfo(repo)
    weights = get_network_path(repo, network)+str(epoch)+'.h5'
    nmod = importlib.import_module('sim.networks.'+network)
    model = nmod.model(dinfo)
    model.load_weights(weights)
    
    print("Creating AV-generator for "+str(testset))
    gen = avdata.AVGenerator(dinfo, testset)
    res = []

    print("Running AV test for "+str(testset))
    per = 0
    for i, (uid, ts, ls, Xs, label) in enumerate(gen):
        if i >= per*len(gen):
            print(str(round(per*100))+'%')
            per += max(0.01, 1.0/len(gen))
        
        ys = np.empty((0,2))
        for x in Xs:
            ys = np.vstack([ys, model.predict(x)])
        res.append((uid,ts,ls,ys,label))

    path = get_sim_path(repo)
    with open(path+network+'-'+testset+'.csv', 'w') as out:
        for (uid,ts,ls,ys,label) in res:
            out.write(str(uid)+";"+str(label)+';'+';'.join([(str(t)+','+str(l)+','+str(y[1])) for t,l,y in zip(ts,ls,ys)])+'\n')

def run_combine(fun, problems, delta):
    accuracy = 0
    T, F = 0, 0
    PT, PF = 0, 0
    TP, FP, TN, FN = 0, 0, 0, 0
    for label,seq in problems:
        pred = fun(seq)
        pred = (pred >= delta)
        accuracy += 1 if pred == label else 0
        if label:
            T += 1
        else:
            F += 1
        if pred:
            PT += 1
            if label:
                TP += 1
            else:
                FP += 1
        else:
            PF += 1
            if label:
                FN += 1
            else:
                TN += 1
    FAR = FN / float(FN+TN) if FN+TN > 0 else 1.0
    return [delta, T, F, PT, PF, TP, FP, TN, FN, accuracy/len(problems), FAR]

def pp(d, line=True):
    [delta, T, F, PT, PF, TP, FP, TN, FN, accuracy, FAR] = d
    if line:
        return "{0:.2f}".format(delta)+';'+';'.join(str(x) for x in [T, F, PT, PF, TP, FP, TN, FN])+';'+"{0:.5f}".format(accuracy)+';'+"{0:.5f}".format(FAR)+"\n"
    else:
        res  = "Split T/F: ("+str(T)+"/"+str(F)+")\n"
        res += "Preds T/F: ("+str(PT)+"/"+str(FP)+")\n"
        res += "Accuracy:  {0:.5f}\n".format(accuracy)
        res += "TP:        "+str(TP)+"\n"
        res += "FP:        "+str(FP)+"\n"
        res += "TN:        "+str(TN)+"\n"
        res += "FN:        "+str(FN)+"\n"
        res += "FAR:       {0:.5}\n".format(FAR)
        res += "Delta:     {0:.2f}\n".format(delta)
    return res

def eval_combine(fun, problems, fname):
    with open(fname, 'w') as f:
        f.write('Delta;True;False;PredTrue;PredFalse;TP;FP;TN;FN;Accuracy;FAR\n')
        results = []
        for delta in np.arange(0.01, 1.0, 0.01):
            d = run_combine(fun, problems, delta)
            f.write(pp(d))
            results.append(d)
        return results

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

def get_sim_path(datarepo):
    config = get_config()
    path = config['path_storage']+'av/'+datarepo+'/similarities/'
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def get_output_path(datarepo, subdir):
    config = get_config()
    path = config['path_storage']+'av/'+datarepo+'/output/'+subdir+'/'
    if not os.path.exists(path):
        os.makedirs(path)
    return path

