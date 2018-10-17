import random
from io import open
import numpy as np

from util.profiles import PROFILES

# streams = 'char', 'words', 'tokens'
def load_data(datafile, dataset="MaCom"):
    res = dict()
    path = "data/"+dataset+"/processed/"
    with open(path+datafile+".csv", 'r', encoding="utf8") as ftxt:
        for l in ftxt:
            l = l.strip().split(";")
            uid = l[0]
            txt = l[1]
            if uid not in res:
                res[uid] = [[],[],[]]
            res[uid][0].append(txt)

    with open(path+datafile+"_words.csv", 'r', encoding="utf8") as fwrd:
        for l in fwrd:
            l = l.strip().split(";")
            uid = l[0]
            words = l[1].split(' ')
            res[uid][1].append(words)
        
    with open(path+datafile+"_pos.csv", 'r', encoding="utf8") as fpos:
        for l in fpos:
            l = l.strip().split(";")
            uid = l[0]
            pos = l[1].split(' ')
            res[uid][2].append(pos)
    
    for auth,[c,w,p] in res.items():
        res[auth] = list(zip(c,w,p))
    
    return res

def generate_stats(datafile, dataset="MaCom"):
    data = load_data(datafile, dataset)
    profile = PROFILES[dataset]
    text = ""
    words = []
    for _,authdata in data.items():
        for txt,wrd,pos in authdata:
            text += txt
            words += wrd
    data = None

    cmap = dict()
    for c in text:
        if c not in cmap:
            cmap[c] = 0
        cmap[c] += 1
    text = None

    wmap = dict()
    for w in words:
        if w not in wmap:
            wmap[w] = 0
        wmap[w] += 1
    words = None

    cmap = list(cmap.items())
    cmap.sort(key=lambda x:-x[1])
    cmap = cmap[:profile["char_map_size"]-1]

    wmap = list(wmap.items())
    wmap.sort(key=lambda x:-x[1])
    wmap = wmap[:profile["word_map_size"]-1]

    path = "data/"+dataset+"/processed/"
    with open(path+'cmap.txt', 'w', encoding="utf8") as f:
        for c in cmap:
            f.write(str(c[0])+";"+str(c[1])+"\n")
    with open(path+'wmap.txt', 'w', encoding="utf8") as f:
        for w in wmap:
            f.write(str(w[0])+";"+str(w[1])+"\n")

def load_stats(dataset="MaCom"):
    cmap, wmap = dict(), dict()
    path = "data/"+dataset+"/processed/"
    with open(path+'cmap.txt', 'r', encoding="utf8") as f:
        for i,l in enumerate(f):
            l = l.split(";")
            cmap[l[0]] = i+1
    with open(path+'wmap.txt', 'r', encoding="utf8") as f:
        for i,l in enumerate(f):
            l = l.split(";")
            wmap[l[0]] = i+1
    return cmap, wmap

def prepare_text(data, cmap, wmap, profile):
    txt, wrds, pos = data
    chars = []
    words = []
    for c in txt:
        chars.append(cmap.get(c, 0))
    chars = chars[:10000]
    while len(chars) < 10000:
        chars.append(0)
    for w in wrds:
        words.append(wmap.get(w, 0))
    words = words[:3000]
    while len(words) < 3000:
        words.append(0)
    return chars, words, []



def get_siamese_set(datafile, dataset="MaCom", formatinput=True):
    profile = PROFILES[dataset]
    authors = list(load_data(datafile, dataset).items())
    authors_processed = []

    cmap, wmap = load_stats(dataset)
    
    for (uid, data) in authors:
        processed = []
        for d in data:
            dproc = prepare_text(d, cmap, wmap, profile)
            processed.append(dproc)

        authors_processed.append((uid, processed))

    authors = None
    
    dataset = []
    for aidx, (author, data) in enumerate(authors_processed):
        for i in range(len(data)):
            for j in range(i+1,len(data)):
                d1 = data[i]
                d2 = data[j]
                dataset.append(((d1[0], d1[1]), (d2[0],d2[1]), 1))
        
                gw = aidx
                while gw == aidx:
                    gw = random.randint(0, len(authors_processed)-1)
                baddata = authors_processed[gw][1]
                godd = data[random.randint(0, len(data)-1)]
                badd = baddata[random.randint(0, len(baddata)-1)]
                dataset.append(((godd[0], godd[1]), (badd[0], badd[1]), 0))

    random.shuffle(dataset)
    
    if formatinput:
        inp = dict()
        inp['known_char_in'] = np.array([x[0][0] for x in dataset])
        inp['known_word_in'] = np.array([x[0][1] for x in dataset])
        inp['unknown_char_in'] = np.array([x[1][0] for x in dataset])
        inp['unknown_word_in'] = np.array([x[1][1] for x in dataset])
        out = {'output':np.array([[1,0] if x[2] else [0,1] for x in dataset])}
        return inp, out

    return dataset
    

