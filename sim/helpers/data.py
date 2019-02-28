import random
import numpy as np
import keras
import configparser
import re
import datetime
from keras.preprocessing import sequence

from . import util

POS_REDUCED_MAP = {'NOUN':1,'VERB':2,'PRON':3,'ADJ':4,'ADP':5,'ADV':6,'PROPN':7,'CONJ':8}

class DataInfo:
    def __init__(self, dataset, load_maps=True):
        dconfig = configparser.ConfigParser()
        print(util.get_data_path(dataset)+"info.ini")
        dconfig.read(util.get_data_path(dataset)+"info.ini")
        dconfig = dconfig['Info']

        self.dataset = dataset

        self.text_max_length = int(dconfig.get('text_max_length', -1))
        self.word_max_length = int(dconfig.get('word_max_length', -1))
        self.pos_max_length  = int(dconfig.get('pos_max_length', self.word_max_length))
        self.pos_sample_prob = float(dconfig.get('pos_sample_prob', 0.1))

        self.char_freq_cutoff = float(dconfig.get('char_freq_cutoff', 0.0))
        self.word_freq_cutoff = float(dconfig.get('word_freq_cutoff', 0.0))

        if load_maps:
            self.cmap, self.wmap, self.pmap = load_stats(dataset)
        else:
            self.cmap, self.wmap, self.pmap = dict(), dict(), dict()

        self._channels = ['char']
        self._batch_size = 32

    def channels(self, channels=None):
        if channels is not None:
            self._channels = channels
        return self._channels

    def batch_size(self, size=None):
        if size is not None:
            self._batch_size = size
        return self._batch_size

    def channel_size(self,channel):
        return {'char':len(self.cmap),
                'word':len(self.wmap),
                'pos':len(self.pmap),
                'pos-reduced':len(POS_REDUCED_MAP),
                'pos-sampled':len(POS_REDUCED_MAP)
                }[channel]+1

    def encode(self, data):
        def _encode(stream, mp, upper=None, df=0):
            res = []
            for s in stream:
                res.append(mp.get(s, df))
            if upper > 0:
                res = res[:upper]
            return res

        assert len(self._channels) == len(data)
        
        res = []
        for i,c in enumerate(self._channels):
            if c == 'char':
                res.append(_encode(data[i], self.cmap, self.text_max_length))
            elif c == 'word':
                res.append(_encode(data[i], self.wmap, self.word_max_length))
            elif c == 'pos':
                res.append(_encode(data[i], self.pmap, self.pos_max_length))
            elif c == 'pos-reduced':
                res.append(_encode(data[i], POS_REDUCED_MAP, self.pos_max_length))
            elif c == 'pos-sampled':
                pos = _encode(data[i], POS_REDUCED_MAP, self.pos_max_length)
                pos_sampled = []
                seql = 5
                while len(pos_sampled) < self.pos_max_length*self.pos_sample_prob:
                    j = random.randint(0,len(pos)-seql)
                    pos_sampled += pos[j:j+seql]
                res.append(pos_sampled)
        return tuple(res)

class SiameseGenerator(keras.utils.Sequence):
    def __init__(self, datainfo, filename, shuffle=True, subsample=None, inclNeg=True, inclPos=True):
        self.datainfo = datainfo
        self.shuffle  = shuffle
        self.subsample = subsample
        self.authors  = []
        self.data     = []
        self.problems = []
        self.inclNeg = inclNeg
        self.inclPos = inclPos
        
        self.get_data(filename)
        self.construct_problems() 
        self.on_epoch_end()
   
    def get_data(self, filename):
        self.data = []
        self.authors = []
        auths = list(load_data(filename, self.datainfo.dataset,
            self.datainfo.channels(), incl_ts=False).items())
        
        for (uid, data) in auths:
            processed = []
            for d in data:
                proc = self.datainfo.encode(d)
                self.data.append(proc)
                processed.append(len(self.data)-1)
            self.authors.append(processed)

    def construct_problems(self):
        self.problems = []
        for aidx, data in enumerate(self.authors):
            for i,pi in enumerate(data):
                for j in range(i+1,len(data)):
                    pj = data[j]
                    if self.inclPos:
                        self.problems.append(((pi,pj), 1))
                
                    if self.inclNeg:
                        a1 = random.randint(0, len(self.authors)-1)
                        a2 = a1
                        while a2 == a1:
                            a2 = random.randint(0,len(self.authors)-1)
                        p1 = random.choice(self.authors[a1])
                        p2 = random.choice(self.authors[a2])
                        self.problems.append(((p1, p2), 0))

        random.shuffle(self.problems)

    def __len__(self):
        return int(np.floor(len(self.indexes) / self.datainfo.batch_size()))

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index*self.datainfo.batch_size():(index+1)*self.datainfo.batch_size()]

        # Find list of IDs
        temp = [self.problems[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(temp)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.problems))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            ssp = self.subsample
            if ssp is not None:
                self.indexes = self.indexes[:int(ssp*len(self.indexes))]

    def __data_generation(self, ids):
        X = dict()
        for cidx,c in enumerate(self.datainfo.channels()):
            known, unknown = self.prep_channel(cidx, ids)
            X['known_'+c+'_in'] = known
            X['unknown_'+c+'_in'] = unknown
        
        y = np.empty((self.datainfo.batch_size()), dtype=int)
        for idx, (_, label) in enumerate(ids):
            y[idx] = label

        # Encode output label: y=0 => [1,0], y=1 => [0,1]
        return X, keras.utils.to_categorical(y, num_classes=2)

    def prep_channel(self, cidx, ids):
        known   = []
        unknown = []
        for ((i,j), _) in ids:
            dati = self.data[i]
            datj = self.data[j]

            known.append(dati[cidx])
            unknown.append(datj[cidx])
       
        kls  = [len(x) for x in known]
        uls  = [len(x) for x in unknown]
        kmin = min(max(1000, min(kls)), max(kls))
        umin = min(max(1000, min(uls)), max(uls))
        known = sequence.pad_sequences(known, value=0, maxlen=kmin, truncating='post', padding='post')
        unknown = sequence.pad_sequences(unknown, value=0, maxlen=umin, truncating='post', padding='post')
        
        return np.array(known), np.array(unknown)

class AVGenerator(keras.utils.Sequence):
    def __init__(self, datainfo, filename):
        self.datainfo = datainfo
        self.authors  = []
        self.data     = []
        self.problems = []
        
        self.get_data(filename)
        self.construct_problems()
   
    def get_data(self, filename):
        self.authors = []
        auths = list(load_data(filename, self.datainfo.dataset,
            self.datainfo.channels(), incl_ts=True).items())
        
        for (uid, data) in auths:
            texts = []
            data.sort(key=lambda x: x[0])
            for d in data:
                ts = d[0]
                ls = len(d[1])
                proc = self.datainfo.encode(d[1:])
                texts.append((ts, ls, proc))
            self.authors.append((uid, texts))

    def construct_problems(self, prob=0.5):
        self.problems = []
        sprob = 1.0/(1.0-prob)-1.0
        for aidx, _ in enumerate(self.authors):
            self.problems.append((aidx,aidx,-1))
            if np.random.rand() < sprob:
                oidx = aidx
                while oidx == aidx:
                    oidx = np.random.randint(0,len(self.authors))
                self.problems.append((aidx,oidx,np.random.randint(0, len(self.authors[oidx]))))

    def __len__(self):
        return len(self.problems)

    def __getitem__(self, index):
        (a1idx,a2idx,tidx) = self.problems[index]
        label = 1 if a1idx == a2idx else 0
        uid     = self.authors[a1idx][0]
        knowns  = self.authors[a1idx][1][:-1]
        unknown = self.authors[a2idx][1][tidx]

        ts = [x[0] for x in knowns]
        ls = [x[1] for x in knowns]
        X = self.__data_generation(knowns, unknown)
        
        return (uid, ts, ls, X, label)

    def __data_generation(self, knowns, unknown):
        X = dict()
        unknown = unknown[2]
        knowns  = [x[2] for x in knowns]
        for cidx,c in enumerate(self.datainfo.channels()):
            k, u = self.prep_channel(cidx, knowns, unknown)
            X['known_'+c+'_in'] = k
            X['unknown_'+c+'_in'] = u
        return X

    def prep_channel(self, cidx, knowns, unknown):
        k, u = [x[cidx] for x in knowns], [unknown[cidx]]*len(knowns)
        
        k = sequence.pad_sequences(k, value=0, padding='post')
        u = sequence.pad_sequences(u, value=0, padding='post')
        
        return np.array(k), np.array(u)

# streams = 'char', 'word', 'pos', 'pos_sampled', 'pos_reduced'
def load_data(datafile, dataset, channels=('char','word','pos'), incl_ts=True):
    res = dict()
    path = util.get_data_path(dataset)+"processed/"
    def load_channel(fname, fun=None):
        chres = dict()
        with open(fname, 'r', encoding="utf8") as chan:
            for l in chan:
                l = l.strip().split(";")
                uid, ts, val = l if len(l) == 3 else (l[0], None, l[1])
                if uid not in chres:
                    chres[uid] = []
                chres[uid].append(val if fun is None else fun(val))
        return chres

    res = dict()
    channels = channels if not incl_ts else ['ts']+list(channels)
    def posfilter(x):
        return list(filter(lambda y: y in POS_REDUCED_MAP, x.split(' ')))
    for c in channels:
        if c == 'ts':
            chres = load_channel(path+datafile+"_ts.csv",
                        fun=lambda x: int(datetime.datetime.strptime(x, '%d-%m-%Y').timestamp()))
        elif c == 'char':
            chres = load_channel(path+datafile+'.csv', fun=util.clean)
        elif c == 'word':
            chres = load_channel(path+datafile+'_words.csv', fun=lambda x: x.split(' '))
        elif c == 'pos':
            chres = load_channel(path+datafile+'_pos.csv', fun=lambda x: x.split(' '))
        elif c == 'pos_reduced':
            chres = load_channel(path+datefile+'_pos.csv', fun=posfilter)
        elif c == 'pos_sampled':
            chres = load_channel(path+datefile+'_pos.csv', fun=posfilter)

        for uid, val in chres.items():
            if uid not in res:
                res[uid] = []
            res[uid].append(val)
    
    for auth,vals in res.items():
        zipped = []
        for i in range(len(vals[0])):
            elem = []
            for d in range(len(vals)):
                elem.append(vals[d][i])
            zipped.append(tuple(elem))
        res[auth] = zipped
    
    return res

def generate_stats(datafile, dataset):
    dinfo = DataInfo(dataset, load_maps=False)
    print("Loading data")
    data = load_data(datafile, dataset)
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
        for _,txt,wrd,pos in authdata:
            txt = re.sub(r'\$PROPN\$', '', txt)
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
    cmap = [x for x in cmap if x[1] > dinfo.char_freq_cutoff*ctot]
    cmap.sort(key=lambda x:-x[1])

    wmap = list(wmap.items())
    wmap = [x for x in wmap if x[1] > dinfo.word_freq_cutoff*wtot]
    wmap.sort(key=lambda x:-x[1])
    
    pmap = list(pmap.items())
    pmap.sort(key=lambda x:-x[1])

    print("Writing maps")
    path = util.get_data_path(dataset)+"processed/"
    with open(path+'cmap.txt', 'w', encoding="utf8") as f:
        for c in cmap:
            ch = c[0] if c[0] != '\n' else '$NL$'
            f.write(str(ch)+";"+str(c[1])+"\n")
    with open(path+'wmap.txt', 'w', encoding="utf8") as f:
        for w in wmap:
            f.write(str(w[0])+";"+str(w[1])+"\n")
    with open(path+'pmap.txt', 'w', encoding="utf8") as f:
        for p in pmap:
            f.write(str(p[0])+";"+str(p[1])+"\n")

def load_stats(dataset="MaCom"):
    cmap, wmap, pmap = dict(), dict(), dict()
    path = util.get_data_path(dataset)+"processed/"
    with open(path+'cmap.txt', 'r', encoding="utf8") as f:
        for i,l in enumerate(f):
            l = l.split(";")
            ch = l[0] if l[0] != '$NL$' else '\n'
            ch = ch if ch != '$SC$' else ';'
            cmap[l[0]] = i+1
        assert len(cmap) > 0
    with open(path+'wmap.txt', 'r', encoding="utf8") as f:
        for i,l in enumerate(f):
            l = l.split(";")
            wmap[l[0]] = i+1
        assert len(wmap) > 0
    with open(path+'pmap.txt', 'r', encoding="utf8") as f:
        for i,l in enumerate(f):
            l = l.split(";")
            pmap[l[0]] = i+1
        assert len(pmap) > 0
    return cmap, wmap, pmap

def info(datafile, datarepo):
    n_authors, n_txt, n_sim, n_av = 0, 0, 0, 0
    avg_txt_author, avg_len_char, avg_len_word = 0.0, 0.0, 0.0

    dinfo = DataInfo(datarepo)
    
    raw = load_data(datafile, dataset=datarepo, channels=('char','word'))
    
    n_authors = len(raw)
    for (a,ls) in raw.items():
        n_txt += len(ls)
        for (t,c,w) in ls:
            avg_len_char += len(c)
            avg_len_word += len(w)

    avg_len_char /= float(n_txt)
    avg_len_word /= float(n_txt)
    avg_txt_author = n_txt / float(n_authors)

    print('#authors = '+str(n_authors))
    print('#texts = '+str(n_txt))
    print('Mean text length (char) = '+str(avg_len_char))
    print('Mean text length (word) = '+str(avg_len_word))
    print('Avg. text pr. author = '+str(avg_txt_author))

    sgen = SiameseGenerator(dinfo,datafile)
    print('#Sim = '+str(len(sgen)*dinfo.batch_size()))

    #agen = AVGenerator(dinfo,datafile)
    #print('#AV = '+str(len(agen)))
