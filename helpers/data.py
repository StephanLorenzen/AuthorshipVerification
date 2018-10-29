import random
from io import open
import numpy as np
import keras

from helpers.profiles import PROFILES


class DataGenerator(keras.utils.Sequence):
    
    def __init__(self, ids, data, batch_size=32, channels=('char', 'word', 'pos'), shuffle=True):
        #self.dim = dim
        self.batch_size = batch_size
        self.data = data
        self.ids = ids 
        self.channels = channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.ids) / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        temp = [self.ids[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(temp)

        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.ids))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, ids):
        cc = 'char' in self.channels
        wc = 'word' in self.channels
        pc = 'pos' in self.channels

        if cc:
            kcc = np.empty((self.batch_size, 10000))
            ucc = np.empty((self.batch_size, 10000))

        if wc:
            uwc = np.empty((self.batch_size, 3000)) 
            kwc = np.empty((self.batch_size, 3000)) 
        
        if pc:
            upc = np.empty((self.batch_size, 3000)) 
            kpc = np.empty((self.batch_size, 3000)) 

        #import pdb; pdb.set_trace()

        for idx, ((i,j), label) in enumerate(ids):
            ai,(kchar, kword, kpos) = self.data[i]
            aj,(uchar, uword, upos) = self.data[j]
            
            if cc:
                kcc[idx,] = kchar
                ucc[idx,] = uchar
            
            if wc:
                kwc[idx,] = kword
                uwc[idx,] = uword
            
            if pc:
                kpc[idx,] = kpos
                upc[idx,] = upos
        
        X = dict()
        if cc:
            X['known_char_in']   = kcc
            X['unknown_char_in'] = ucc
        if wc:
            X['known_word_in']   = kwc
            X['unknown_word_in'] = uwc
        if pc:
            X['known_pos_in']    = kpc
            X['unknown_pos_in']  = upc

        y = np.empty((self.batch_size), dtype=int)
        for idx, (_, label) in enumerate(ids):
            y[idx] = label

        return X, keras.utils.to_categorical(y, num_classes=2)


# streams = 'char', 'words', 'tokens'
def load_data(datafile, dataset="MaCom", channels=('char','word','pos')):
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

    if 'word' in channels:
        with open(path+datafile+"_words.csv", 'r', encoding="utf8") as fwrd:
            for l in fwrd:
                l = l.strip().split(";")
                uid = l[0]
                words = l[1].split(' ')
                res[uid][1].append(words)
    else:
        for uid,v in res.items():
            for _ in range(len(v[0])):
                v[1].append([])

    if 'pos' in channels:
        with open(path+datafile+"_pos.csv", 'r', encoding="utf8") as fpos:
            for l in fpos:
                l = l.strip().split(";")
                uid = l[0]
                pos = l[1].split(' ')
                res[uid][2].append(pos)
    else:
        for uid,v in res.items():
            for _ in range(len(v[0])):
                v[2].append([])
    
    for auth,[c,w,p] in res.items():
        res[auth] = list(zip(c,w,p))
    
    return res

def generate_stats(datafile, dataset="MaCom"):
    data = load_data(datafile, dataset)
    profile = PROFILES[dataset]
    text = ""
    words = []
    poss = []
    for _,authdata in data.items():
        for txt,wrd,pos in authdata:
            text += txt
            words += wrd
            poss += pos
    data = None

    cmap = dict()
    for c in text:
        if c not in cmap:
            cmap[c] = 0
        cmap[c] += 1

    wmap = dict()
    for w in words:
        if w not in wmap:
            wmap[w] = 0
        wmap[w] += 1
    
    pmap = dict()
    for p in poss:
        if p not in pmap:
            pmap[p] = 0
        pmap[p] += 1

    cmap = list(cmap.items())
    cmap.sort(key=lambda x:-x[1])
    cmap = cmap[:profile["char_map_size"]-1]

    wmap = list(wmap.items())
    wmap.sort(key=lambda x:-x[1])
    wmap = wmap[:profile["word_map_size"]-1]
    
    pmap = list(pmap.items())
    pmap.sort(key=lambda x:-x[1])
    pmap = pmap[:profile["pos_map_size"]-1]

    path = "data/"+dataset+"/processed/"
    with open(path+'cmap.txt', 'w', encoding="utf8") as f:
        for c in cmap:
            f.write(str(c[0])+";"+str(c[1])+"\n")
    with open(path+'wmap.txt', 'w', encoding="utf8") as f:
        for w in wmap:
            f.write(str(w[0])+";"+str(w[1])+"\n")
    with open(path+'pmap.txt', 'w', encoding="utf8") as f:
        for p in pmap:
            f.write(str(p[0])+";"+str(p[1])+"\n")

def load_stats(dataset="MaCom"):
    cmap, wmap, pmap = dict(), dict(), dict()
    path = "data/"+dataset+"/processed/"
    with open(path+'cmap.txt', 'r', encoding="utf8") as f:
        for i,l in enumerate(f):
            l = l.split(";")
            cmap[l[0]] = i+1
    with open(path+'wmap.txt', 'r', encoding="utf8") as f:
        for i,l in enumerate(f):
            l = l.split(";")
            wmap[l[0]] = i+1
    with open(path+'pmap.txt', 'w', encoding="utf8") as f:
        for p in pmap:
            f.write(str(p[0])+";"+str(p[1])+"\n")
    return cmap, wmap, pmap

def prepare_text(data, cmap, wmap, pmap, profile):
    txt, wrds, pos = data
    chars = []
    words = []
    poss  = []
    # Chars
    for c in txt:
        chars.append(cmap.get(c, 0))
    chars = chars[:10000]
    while len(chars) < 10000:
        chars.append(0)
    # Words
    for w in wrds:
        words.append(wmap.get(w, 0))
    words = words[:3000]
    while len(words) < 3000:
        words.append(0)
    # POS tags
    for p in pos:
        poss.append(pmap.get(p, 0))
    poss = poss[:3000]
    while len(poss) < 3000:
        poss.append(0)
    return chars, words, poss

def get_siamese_set(datafile, dataset="MaCom", formatinput=True):
    profile = PROFILES[dataset]
    authors = list(load_data(datafile, dataset).items())
    authors_processed = []

    cmap, wmap, pmap = load_stats(dataset)
    
    for (uid, data) in authors:
        processed = []
        for d in data:
            dproc = prepare_text(d, cmap, wmap, pmap, profile)
            processed.append(dproc)

        authors_processed.append((uid, processed))

    authors = None
    
    dataset = []
    for aidx, (author, data) in enumerate(authors_processed):
        for i in range(len(data)):
            for j in range(i+1,len(data)):
                d1 = data[i]
                d2 = data[j]
                dataset.append((tuple(d1), tuple(d2), 1))
        
                gw = aidx
                while gw == aidx:
                    gw = random.randint(0, len(authors_processed)-1)
                baddata = authors_processed[gw][1]
                godd = data[random.randint(0, len(data)-1)]
                badd = baddata[random.randint(0, len(baddata)-1)]
                dataset.append((tuple(godd), tuple(badd), 0))

    random.shuffle(dataset)
    
    if formatinput:
        inp = dict()
        inp['known_char_in'] = np.array([x[0][0] for x in dataset])
        inp['known_word_in'] = np.array([x[0][1] for x in dataset])
        inp['known_pos_in']  = np.array([x[0][2] for x in dataset])
        inp['unknown_char_in'] = np.array([x[1][0] for x in dataset])
        inp['unknown_word_in'] = np.array([x[1][1] for x in dataset])
        inp['unknown_pos_in']  = np.array([x[1][2] for x in dataset])
        out = {'output':np.array([[1,0] if x[2] else [0,1] for x in dataset])}
        return inp, out

    return dataset
    
def get_siamese_generator(datafile, dataset="MaCom", channels=('char','word','pos'), batch_size=32):
    profile = PROFILES[dataset]
    authors = list(load_data(datafile, dataset, channels).items())
    authors_processed = []

    cmap, wmap, pmap = load_stats(dataset)
   
    alltexts = []

    for (uid, data) in authors:
        processed = []
        for d in data:
            dproc = prepare_text(d, cmap, wmap, pmap, profile)
            alltexts.append((uid, dproc))
            processed.append(dproc)

        authors_processed.append((uid, processed))

    authors = None
    
    ids = []
    for aidx, (author, data) in enumerate(authors_processed):
        for i in range(len(data)):
            for j in range(i+1,len(data)):
                ids.append(((i,j), 1))
        
                gw = aidx
                while gw == aidx:
                    gw = random.randint(0, len(authors_processed)-1)
                ids.append(((aidx, gw), 0))

    random.shuffle(ids)
    
    return DataGenerator(ids, alltexts, channels=channels, batch_size=batch_size)

