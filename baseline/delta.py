from sklearn.svm import SVC
import random
import numpy as np

import helpers.data as avdata

datarepo = 'PAN13'
train = 'train'
test = 'test'
numfeat = 150
checkK = 10

dinfo = avdata.DataInfo(datarepo)

def stats(ls):
    mp = dict()
    for _,t in ls:
        for w in t:
            if w not in mp:
                mp[w] = 0
            mp[w] += 1
    top = sorted(list(mp.items()), key=lambda x: -x[1])
    return [x[0] for x in top[:numfeat]]

def encode(t, feats):
    mp = dict([(f,0) for f in feats])
    for w in t:
        if w in mp:
            mp[w] += 1
    res = []
    for f in feats:
        res.append(mp[f])
    return np.array(res)

authors = avdata.load_data(train, datarepo, ('word',), False)
authors = [v for _,v in authors.items()]
alltexts = []
for a,v in enumerate(authors):
    alltexts += [(a,t[0]) for t in v]
    checkK = min((len(v)-1)*2,checkK)
authors = dict()
for i,(a,_) in enumerate(alltexts):
    if a not in authors:
        authors[a] = []
    authors[a].append(i)

feats = stats(alltexts)

train = [(a,encode(t, feats)) for a,t in alltexts]

def dist(t1, t2):
    return np.sum(np.abs(t1-t2))
def predict(unknown, author, other, ks=[3]):
    als = []
    for _,t in author:
        als.append((1,dist(unknown[1],t)))
    for _,t in other:
        als.append((0,dist(unknown[1],t)))
    als.sort(key=lambda x: x[1])
    res = [0.0]*max(ks)
    for k in range(max(ks)):
        res[k] = res[k-1] if k > 0 else 0.0
        res[k] += als[k][0]
        res[k] = round(res[k]/(k+1)+0.00001)
    ret = []
    for k in ks:
        ret.append(res[k-1])
    return ret
def eval_delta(problems, data, ks=[3]):
    P = [[0]*6 for _ in ks]
    for (authorIdx,otherIdx) in problems:
        author = [data[i] for i in authorIdx[:-1]]
        other  = [data[i] for i in otherIdx[:-1]]
        good   = data[authorIdx[-1]]
        bad    = data[otherIdx[-1]]
        ps1 = predict(good, author, other, ks)
        ps0 = predict(bad, author, other, ks)
        for i,k in enumerate(ks):
            stat = P[i] # PF, PT, TP, FP, TN, FN
            p1 = ps1[i]
            p0 = ps0[i]
            stat[p1] += 1 # PF/PT
            stat[p0] += 1 # PF/PT
            if p1 == 1:
                stat[2] += 1 # TP
            else:
                stat[5] += 1 # FN
            if p0 == 0:
                stat[4] += 1 # TN
            else:
                stat[3] += 1 # FP
    for stat in P:
        stat.append(float(stat[2]+stat[4])/(2*len(problems))) # Accuracy = (TP+TN) / #problems
        stat.append(float(stat[5])/float(stat[5]+stat[4])) # FAR = (FN/(TN+FN))
    return P

print("Selecting K")
problems = []
for a,v in authors.items():
    n = len(v)
    other = random.sample([i for i,(o,_) in enumerate(alltexts) if o!=a], n)
    problems.append((v,other))
    
ks = list(range(1,checkK))
res = eval_delta(problems, train, ks)
best = 0
for i,r in enumerate(res):
    print(str(ks[i])+" ==> "+str(r))
    if r[-2] > res[best][-2]:
        best = i
print("Best = "+str(ks[best])+" : "+str(res[best]))
print("")

print('Testing...')
authors = avdata.load_data(test, datarepo, ('word',), False)
authors = [v for _,v in authors.items()]
alltexts = []
for a,v in enumerate(authors):
    alltexts += [(a,t[0]) for t in v]
authors = dict()
for i,(a,_) in enumerate(alltexts):
    if a not in authors:
        authors[a] = []
    authors[a].append(i)

test = [(a,encode(t, feats)) for a,t in alltexts]

problems = []
for a,v in authors.items():
    n = len(v)
    other = random.sample([i for i,(o,_) in enumerate(alltexts) if o!=a], n)
    problems.append((v,other))

k = [ks[best]]
res = eval_delta(problems, test, k)
print("Result")
res = res[0] # PF, PT, TP, FP, TN, FN, acc, FAR
print("=> Accuracy = "+str(res[-2]))
print("=> Balance = 50/50")
print("=> Preds (T/F) = ("+str(res[1])+"/"+str(res[0])+")")
print("=> TP  = "+str(res[2]))
print("=> FP  = "+str(res[3]))
print("=> TN  = "+str(res[4]))
print("=> FN  = "+str(res[5]))
print("=> FAR = "+str(res[-1]))


