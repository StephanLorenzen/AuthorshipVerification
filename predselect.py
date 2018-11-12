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

    with open(fname) as f:
        problems = []
        for l in f:
            l = l.strip().split(';')
            label = (l[0]=='1')
            preds = []
            for p in l[1:]:
                time,score = p.split(',')
                preds.append((int(time),float(score)))
            problems.append((label, preds))

    def eval_combine(fun, problems, delta):
        accuracy = 0
        for label,seq in problems:
            pred = fun(seq)
            pred = (pred >= delta)
            accuracy += 1 if pred == label else 0
        return float(accuracy) / len(problems)

    delta = 0.5

    print("Evaluating min")
    acc = eval_combine(combine.cmin, problems, delta)
    print(" => Accuracy = "+str(acc))

    print("Evaluating max")
    acc = eval_combine(combine.cmax, problems, delta)
    print(" => Accuracy = "+str(acc))

    print("Evaluating exp")
    for lamb in range(1,11):
        acc = eval_combine(lambda seq: combine.exponential(seq, lamb/10), problems, delta)
        print("Lambda = "+str(lamb/10)+" => Accuracy = "+str(acc))


