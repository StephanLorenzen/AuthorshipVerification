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
    parser.add_argument('-c', '--combine', metavar='FUNC', choices=['max','min','uniform','exp'],
        type=str, default=None, help='Combine function to use. If not set, selection will run.')
    parser.add_argument('-l', '--lamb', metavar='LAMBDA', type=float, default=1.0, help='Lambda for exp-function.')
    parser.add_argument('-t', '--delta', metavar='DELTA', type=float, default=None, help='Threshold.')

    args = parser.parse_args()
    repo = args.datarepo
    trainset = args.TRAINSET
    func = args.combine
    lamb = args.lamb
    delta = args.delta

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
                time,length,score = p.split(',')
                preds.append((int(time),int(length),float(score)))
            problems.append((label, preds))

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

    if func is not None:
        print("Testing with "+str(func)+(", lambda = "+str(lamb) if func=='exp' else '')+", delta = "+str(delta))
        fun = lambda seq: combine.exponential(seq, lamb)
        if func == 'max':
            fun = combine.max
        elif func == 'min':
            fun = combine.min
        elif func == 'uniform':
            fun = combine.uniform

        if delta is None:
            # ROC curve
            eval_combine(fun, problems, outdir+'roc.csv')
        else:
            r = run_combine(fun, problems, delta)
            print(pp(r, line=False))
            methodstr = func + ("{0:.01f}".format(lamb) if func=='exp' else '')
            with open(outdir+'testresult.csv', 'w') as f:
                f.write('Method;Delta;True;False;PredTrue;PredFalse;TP;FP;TN;FN;Accuracy;FAR\n')
                f.write(methodstr+';'+pp(r))
    else:
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
        res = eval_combine(combine.uniform, problems, outdir+'uniform.csv')
        for r in res:
            results.append(['uniform']+r)

        print("Evaluating exp")
        for lt in np.arange(0.00, 0.21, 0.01):
            for ll in np.arange(0.00, 0.11, 0.01):
                mname = 'exp-{0:.2f}-{1:.2f}'.format(lt,ll)
                res = eval_combine(lambda seq: combine.exponential(seq, lt, ll),
                        problems, outdir+mname+'.csv')
                for r in res:
                    results.append([mname]+r)

        print("Evaluating majority")
        res = eval_combine(combine.majority, problems, outdir+'majority.csv')
        for r in res:
            results.append(['majority']+r)

        with open(outdir+'summary.csv', 'w') as f:
            f.write('Threshold;Method;Delta;True;False;PredTrue;PredFalse;TP;FP;TN;FN;Accuracy;FAR\n')
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
                    f.write("{0:.2f}".format(threshold)+';'+best[0]+';'+pp(best[1:]))
