import random
from io import open
import numpy as np
import keras

from helpers.profiles import PROFILES


CHAR_UPPER = 10000
WORD_UPPER = 3000
POS_UPPER  = WORD_UPPER

class DataGenerator(keras.utils.Sequence):
    
    def __init__(self, ids, data, batch_size=32, channels=(('char',0), ('word',0), ('pos',0)),
            shuffle=True):
        #self.dim = dim
        self.batch_size = batch_size
        self.data = data
        self.ids = ids 
        self.channels = [x[0] for x in channels]
        self.mapsize  = [x[1] for x in channels]
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

    def channel_size(self,channel):
        for c,size in zip(self.channels,self.mapsize):
            if c == channel:
                return size+1
        print("Channel not included in generator...")

    def __data_generation(self, ids):
        X = dict()
        for c in self.channels:
            known, unknown = self.prep_channel(c, ids)
            X['known_'+c+'_in'] = known
            X['unknown_'+c+'_in'] = unknown
        
        y = np.empty((self.batch_size), dtype=int)
        for idx, (_, label) in enumerate(ids):
            y[idx] = label

        return X, keras.utils.to_categorical(y, num_classes=2)

    def prep_channel(self, channel, ids):
        cidx = {'char':0,'word':1,'pos':2}[channel]
        known   = []
        unknown = []
        for ((i,j), _) in ids:
            ai,dati = self.data[i]
            aj,datj = self.data[j]

            known.append(dati[cidx])
            unknown.append(datj[cidx])
        
        mx = max([len(x) for x in known]+[len(x) for x in unknown])
        
        known = [pad(x, mx) for x in known]
        unknown = [pad(x, mx) for x in unknown]
        return np.array(known), np.array(unknown)

def pad(x, l):
    while len(x) < l:
        x.append(0)
    return x

# streams = 'char', 'words', 'tokens'
def load_data(datafile, dataset="MaCom", channels=('char','word','pos')):
    res = dict()
    path = "data/"+dataset+"/processed/"
    with open(path+datafile+".csv", 'r', encoding="utf8") as ftxt:
        for l in ftxt:
            l = l.strip().split(";")
            uid = l[0]
            txt = l[-1]
            if uid not in res:
                res[uid] = [[],[],[]]
            res[uid][0].append(txt)

    if 'word' in channels:
        with open(path+datafile+"_words.csv", 'r', encoding="utf8") as fwrd:
            for l in fwrd:
                l = l.strip().split(";")
                uid = l[0]
                words = l[-1].split(' ')
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
                pos = l[-1].split(' ')
                res[uid][2].append(pos)
    else:
        for uid,v in res.items():
            for _ in range(len(v[0])):
                v[2].append([])
    
    for auth,[c,w,p] in res.items():
        res[auth] = list(zip(c,w,p))
    
    return res

def generate_stats(datafile, dataset="MaCom"):
    print("Loading data")
    data = load_data(datafile, dataset)
    profile = PROFILES[dataset]
    print("Creating channels")
    
    cmap = dict()
    wmap = dict()
    pmap = dict()
    print("Creating maps")
    l = len(data.items())
    per = 0
    ctot = 0
    wtot = 0
    for i,(_,authdata) in enumerate(data.items()):
        if float(i)/float(l)*100 > per:
            print(str(per)+"%")
            per+=10
        for txt,wrd,pos in authdata:
            ctot += len(txt)
            wtot += len(wrd)
            for c in txt:
                if c not in cmap:
                    cmap[c] = 0
                cmap[c] += 1
            for w in wrd:
                if w not in wmap:
                    wmap[w] = 0
                wmap[w] += 1
            for p in pos:
                if p not in pmap:
                    pmap[p] = 0
                pmap[p] += 1
    data = None

    print("Post processing")
    cmap = list(cmap.items())
    cmap.sort(key=lambda x:-x[1])
    cmap = [x for x in cmap if x[1] > profile["char_freq_cutoff"]*ctot]
    #cmap = cmap[:profile["char_map_size"]-1]

    wmap = list(wmap.items())
    wmap.sort(key=lambda x:-x[1])
    wmap = [x for x in wmap if x[1] > profile["word_freq_cutoff"]*wtot]
    #wmap = wmap[:profile["word_map_size"]-1]
    
    pmap = list(pmap.items())
    pmap.sort(key=lambda x:-x[1])
    pmap = pmap[:profile["pos_map_size"]-1]

    print("Wrtining maps")
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
    chars = chars[:CHAR_UPPER]
    #while len(chars) < 10000:
    #    chars.append(0)
    # Words
    for w in wrds:
        words.append(wmap.get(w, 0))
    words = words[:WORD_UPPER]
    #while len(words) < 3000:
    #    words.append(0)
    # POS tags
    for p in pos:
        poss.append(pmap.get(p, 0))
    poss = poss[:POS_UPPER]
    #while len(poss) < 3000:
    #    poss.append(0)
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
    
    smap = {'char':len(cmap), 'word':len(wmap), 'pos':len(pmap)}
    channels = [(x,smap[x]) for x in channels]
    return DataGenerator(ids, alltexts, channels=channels, batch_size=batch_size)

