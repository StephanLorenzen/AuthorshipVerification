from sklearn.svm import SVC
import random
import numpy as np

import helpers.data as avdata

datarepo = 'PAN13'
train = 'test'
test = 'train'
numfeat = 150
checkK = 8

USE_AI_MODE = False
USE_AI_MODE = True 

authors = avdata.load_data(train, datarepo, ('word',), False)
authors = [[x[0] for x in v] for _,v in authors.items()]
alltxt  = []
for a,v in enumerate(authors):
    for t in v:
        alltxt.append((a,t))

# Compute word freq in db
mp = dict()
for _,t in alltxt:
    for w in t:
        w = w.lower()
        if w not in mp:
            mp[w] = 0
        mp[w] += 1
top = sorted(list(mp.items()), key=lambda x: -x[1])
FEATURES = [x[0] for x in top[:numfeat]]

def freq(txt, mp):
    freqs = dict([(w,0) for w in mp])
    for w in txt:
        w = w.lower()
        if w in freqs:
            freqs[w] += 1
    return np.array([freqs[w] for w in mp])

def meanAndStd(txts, ws):
    mean = np.zeros(len(ws))
    for t in txts:
        mean += freq(t,ws)
    mean /= len(txts)
    
    var = np.zeros(len(ws))
    for t in txts:
        tmp = (freq(t,ws)-mean)
        var += np.multiply(tmp,tmp)
    var /= len(txts)
    var = np.sqrt(var)
    
    return mean,var

DBMEAN, DBSTD = meanAndStd([x[1] for x in alltxt], FEATURES)

authormeta = []
for i,v in enumerate(authors):
    amean,astd = meanAndStd(v[:-1], FEATURES)
    az = (amean - DBMEAN) / DBSTD
    authormeta.append(az)

# Create problem instances
def get_problems(authors):
    probs = []
    for a, txts in enumerate(authors):
        good = txts[-1]
        txts = txts[:-1]
        bada = a
        while bada == a:
            bada = random.randint(0,len(authors)-1)
        bad  = random.choice(authors[bada])
        probs.append((a, good, bad))
    return probs

problems = get_problems(authors)

def evaluate(delta, problems, authormeta):
    T,F,PT,PF,TN,FN,TP,FP = 0,0,0,0,0,0,0,0
    for (a, good, bad) in problems:
        az = authormeta[a]
        fgood = freq(good, FEATURES)
        zgood = (fgood-DBMEAN) / DBSTD
        fbad  = freq(bad, FEATURES)
        zbad  = (fbad-DBMEAN) / DBSTD
        
        goodans = np.sum(np.abs(az-zgood)) / len(FEATURES)
        badans  = np.sum(np.abs(az-zbad))  / len(FEATURES)

        goodans = (goodans < delta)
        badans  = (badans < delta)
        
        T += 1
        F += 1
        
        if goodans:
            PT += 1
            TP += 1
        else:
            PF += 1
            FN += 1
        if badans:
            PT += 1
            FP += 1
        else:
            PF += 1
            TN += 1
    
    Acc = (TP+TN) / float(T+F)
    FAR = FN / float(TN+FN) if TN+FN > 0 else 0.0
    return (T,F,PT,PF,TN,FN,TP,FP,Acc,FAR)

def evaluateAI(k, problems, authormeta):
    T,F,PT,PF,TN,FN,TP,FP = 0,0,0,0,0,0,0,0
    for (a, good, bad) in problems:
        az  = authormeta[a]
        ozs = random.sample(authormeta[:a]+authormeta[a+1:], k)
        fgood = freq(good, FEATURES)
        zgood = (fgood-DBMEAN) / DBSTD
        fbad  = freq(bad, FEATURES)
        zbad  = (fbad-DBMEAN) / DBSTD
        
        gcomps = []
        bcomps = []
        for oz in ozs:
            gcomps.append(np.sum(np.abs(oz-zgood)) / len(FEATURES))
            bcomps.append(np.sum(np.abs(oz-zbad))  / len(FEATURES))

        goodans = np.sum(np.abs(az-zgood)) / len(FEATURES)
        badans  = np.sum(np.abs(az-zbad))  / len(FEATURES)
         
        goodans = (goodans < min(gcomps))
        badans  = (badans < min(bcomps))
        
        T += 1
        F += 1
        
        if goodans:
            PT += 1
            TP += 1
        else:
            PF += 1
            FN += 1
        if badans:
            PT += 1
            FP += 1
        else:
            PF += 1
            TN += 1
    
    Acc = (TP+TN) / float(T+F)
    FAR = FN / float(TN+FN) if TN+FN > 0 else 0.0
    return (T,F,PT,PF,TN,FN,TP,FP,Acc,FAR)


if USE_AI_MODE:
    best = 0.0
    bestacc = 0.0
    with open('baseline/delta_train_identification.csv', 'w') as f:
        f.write('k;T;F;PT;PF;TN;FN;TP;FP:Accuracy;FAR\n');
        for k in range(1,10):
            out = evaluateAI(k, problems, authormeta)
            f.write(str(k)+';'+';'.join([str(x) for x in out])+'\n')
            (T,F,PT,PF,TN,FN,TP,FP,Acc,FAR) = out
            if Acc > bestacc:
                bestacc = Acc
                best = k
    
    print('Best k found = '+str(best)+", acc = "+str(bestacc))
else:
    best = 0.0
    bestacc = 0.0
    with open('baseline/delta_train.csv', 'w') as f:
        f.write('delta;T;F;PT;PF;TN;FN;TP;FP:Accuracy;FAR\n');
        for delta in np.arange(0.1,5.1,0.1):
            out = evaluate(delta, problems, authormeta)
            f.write(str(delta)+';'+';'.join([str(x) for x in out])+'\n')
            (T,F,PT,PF,TN,FN,TP,FP,Acc,FAR) = out
            if Acc > bestacc:
                bestacc = Acc
                best = delta
    
    print('Best delta found = '+str(best)+", acc = "+str(bestacc))

authors = avdata.load_data(test, datarepo, ('word',), False)
authors = [[x[0] for x in v] for _,v in authors.items()]
alltxt  = []
for a,v in enumerate(authors):
    for t in v:
        alltxt.append((a,t))

authormeta = []
for i,v in enumerate(authors):
    amean,astd = meanAndStd(v[:-1], FEATURES)
    az = (amean - DBMEAN) / DBSTD
    authormeta.append(az)
    
problems = get_problems(authors)

if USE_AI_MODE:
    (T,F,PT,PF,TN,FN,TP,FP,Acc,FAR) = evaluateAI(best, problems, authormeta)
else:
    (T,F,PT,PF,TN,FN,TP,FP,Acc,FAR) = evaluate(best, problems, authormeta)

print("Result")
print("=> Accuracy = "+str(Acc))
print("=> Balance = 50/50")
print("=> Preds (T/F) = ("+str(T)+"/"+str(F)+")")
print("=> TP  = "+str(TP))
print("=> FP  = "+str(FP))
print("=> TN  = "+str(TN))
print("=> FN  = "+str(FN))
print("=> FAR = "+str(FAR))
