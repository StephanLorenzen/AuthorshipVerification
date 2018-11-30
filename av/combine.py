# External imports
import numpy as np
import os

import argparse
import importlib

# Local imports
from .helpers import data as avdata
from .helpers import util as util
from .helpers import combine as combine

def train(args):
    repo = args.datarepo
    network = args.NETWORK
    epoch = args.epoch
    rc    = args.recompute
    trainset = args.TRAINSET

    simset = network+'-'+trainset
    simfile = util.get_sim_path(repo)+simset+'.csv'
    
    if not os.path.isfile(simfile) or rc:
        print("Computing similarities...")
        util.compute_similarities(repo, trainset, network, epoch)
    
    outdir  = util.get_output_path(repo, simset)

    print("Loading network output ("+simset+")")
    problems = avdata.load_similarities(repo, network, trainset)
     
    results = []
    print("Evaluating min")
    res = util.eval_combine(combine.cmin, problems, outdir+'min.csv')
    for r in res:
        results.append(['min']+r)

    print("Evaluating max")
    res = util.eval_combine(combine.cmax, problems, outdir+'max.csv')
    for r in res:
        results.append(['max']+r)
        
    print("Evaluating uniform")
    res = util.eval_combine(combine.uniform, problems, outdir+'uniform.csv')
    for r in res:
        results.append(['uniform']+r)

    print("Evaluating exp")
    for lt in np.arange(0.00, 0.21, 0.01):
        for ll in np.arange(0.00, 0.11, 0.01):
            mname = 'exp-{0:.2f}-{1:.2f}'.format(lt,ll)
            res = util.eval_combine(lambda seq: combine.exponential(seq, lt, ll),
                    problems, outdir+mname+'.csv')
            for r in res:
                results.append([mname]+r)

    print("Evaluating majority")
    res = util.eval_combine(combine.majority, problems, outdir+'majority.csv')
    for r in res:
        results.append(['majority']+r)

    with open(outdir+'summary.csv', 'w') as f:
        f.write('Threshold;Method;Delta;True;False;PredTrue;PredFalse;TP;FP;TN;FN;Accuracy;FAR\n')
        prev = None
        for threshold in np.arange(0.05,1.01,0.05):
            bestacc = 0.0
            best = None
            for r in results:
                [method, delta, T, F, PT, PF, TP, FP, TN, FN, acc, far] = r
                if far < threshold:
                    if acc > bestacc:
                        bestacc = acc
                        best = r
            if best is not None:
                f.write("{0:.2f}".format(threshold)+';'+best[0]+';'+util.pp(best[1:]))
                if best != prev:
                    print('#### Best result for threshold = {0:.2f}'.format(threshold))
                    print("Strategy:  "+str(best[0]))
                    print(util.pp(best[1:], False))
                prev = best

def test(args):
    repo = args.datarepo
    network = args.NETWORK
    epoch = args.epoch
    rc    = args.recompute
    lamb = args.lamb
    func = args.COMBINE
    delta = args.delta
    trainset = args.TESTSET
    
    simset = network+'-'+trainset
    simfile = util.get_sim_path(repo)+simset+'.csv'
    
    if not os.path.isfile(simfile) or rc:
        print("Computing similarities...")
        util.compute_similarities(repo, trainset, network, epoch)
    
    outdir  = util.get_output_path(repo, simset)
    
    print("Loading network output ("+simset+")")
    problems = avdata.load_similarities(repo, network, trainset)
    
    fun = combine.get_fun(func, lamb)

    if delta is None:
        # ROC curve
        print("Generating ROC curve for "+str(func)
                +(", lambda = "+str(lamb) if func=='exp' else ''))
        util.eval_combine(fun, problems, outdir+'test-result-roc.csv')
    else:
        print("Testing with "+str(func)
                +(", lambda = "+str(lamb) if func=='exp' else '')+", delta = "+str(delta))
        r = util.run_combine(fun, problems, delta)
        print(util.pp(r, line=False))
        methodstr = func + ("{0:.2f}".format(lamb) if func=='exp' else '')
        with open(outdir+'test-result.csv', 'w') as f:
            f.write('Method;Delta;True;False;PredTrue;PredFalse;TP;FP;TN;FN;Accuracy;FAR\n')
            f.write(methodstr+';'+util.pp(r))

