# External imports
import numpy as np
import os

import argparse
import importlib

# Local imports
import helpers.data as avdata
import helpers.util as util
import helpers.combine as combine

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Selecting best combination function')
    parser.add_argument('-d', '--datarepo', metavar='DATAREPO', type=str,
        choices=['PAN13', 'MaCom'], default='MaCom',
        help='Data repository to use (default MaCom).')
    parser.add_argument('TRAINSET', type=str, help='Train set file, computed by test.py.')

    args = parser.parse_args()
    repo = args.datarepo
    trainset = args.TRAINSET

    dinfo = avdata.DataInfo(repo)
    
    fname = 'predsys/'+repo+'/'+trainset+'.csv'
    print("Loading network output ("+fname+")")
    outdir = 'predsys/'+repo+'/'+trainset+'/'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    with open(fname) as f:
        problems = []
        for l in f:
            l = l.strip().split(';')
            label = (l[1]=='1')
            preds = []
            for p in l[2:]:
                time,score = p.split(',')
                preds.append((int(time),float(score)))
            problems.append((label, preds))

    def eval_combine(fun, problems, fname):
        with open(fname, 'w') as f:
            f.write('Delta;True;False;PredTrue;PredFalse;TP;FP;TN;FN;Accuracy\n')
            results = []
            for delta in np.arange(0.01, 1.0, 0.01):
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
                result = ["{0:.2f}".format(delta), T, F, PT, PF, TP, FP, TN, FN, float(accuracy) / len(problems)]
                f.write(';'.join([str(x) for x in result])+'\n')
                results.append(result)
            return results

    results = []
    print("Evaluating min")
    res = eval_combine(combine.cmin, problems, outdir+'min.csv')
    for r in res:
        results.append(['min']+r)

    print("Evaluating max")
    res = eval_combine(combine.cmax, problems, outdir+'max.csv')
    for r in res:
        results.append(['max']+r)
    
    print("Evaluating uniform")
    res = eval_combine(combine.uniform, problems, outdir+'max.csv')
    for r in res:
        results.append(['uniform']+r)

    print("Evaluating exp")
    for lamb in np.arange(0.1, 2.0, 0.1):
        res = eval_combine(lambda seq: combine.exponential(seq, lamb/10), problems, outdir+'exp-'+"{0:.1f}".format(lamb)+'.csv')
        for r in res:
            results.append(['exp'+'{0:.1f}'.format(lamb)]+r)

    with open(outdir+'summary.csv', 'w') as f:
        f.write('Threshold;Method;Delta;True;False;PredTrue;PredFalse;TP;FP;TN;FN;Accuracy;FalseAccusationRate\n')
        for threshold in np.arange(0.1,1.01,0.1):
            bestacc = 0.0
            best = None
            for r in results:
                TN = r[8]
                FN = r[9]
                far = FN / float(TN+FN) if TN+FN > 0 else 1
                acc = r[-1]
                if far < threshold:
                    if acc > bestacc:
                        bestacc = acc
                        best = r
            if best is not None:
                TN = best[8]
                FN = best[9]
                far = FN / float(TN+FN) if TN+FN > 0 else 1
                f.write("{0:.5f}".format(threshold)+';'+';'.join([str(x) for x in best])+";"+"{0:.5f}".format(far)+"\n")
