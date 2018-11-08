import random
import helpers.util as util
import numpy as np
import keras
import configparser

from keras.preprocessing import sequence

class DataInfo:
    def __init__(self, dataset):
        config = configparser.ConfigParser()
        config.read('data/'+str(dataset)+"/info.ini")
        config = config['Info']

        self.dataset = dataset

        self.text_max_length = int(config.get('text_max_length', -1))
        self.word_max_length = int(config.get('word_max_length', -1))
        self.pos_max_length  = int(config.get('pos_max_length', self.word_max_length))

        self.char_freq_cutoff = float(config.get('char_freq_cutoff', 0.0))
        self.word_freq_cutoff = float(config.get('word_freq_cutoff', 0.0))

        self.cmap, self.wmap, self.pmap = load_stats(dataset)

        self._channels = []
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
        return {'char':len(self.cmap),'word':len(self.wmap),'pos':len(self.pmap)}[channel]+1

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
        
        return tuple(res)

class DataGenerator(keras.utils.Sequence):
    def __init__(self, datainfo, filename, shuffle=True):
        self.datainfo = datainfo
        self.shuffle  = shuffle
        self.authors  = []
        self.data     = []
        self.problems = []
        
        self.get_data(filename)
        self.construct_problems()
        #import pdb; pdb.set_trace() 
        self.on_epoch_end()
   
    def get_data(self, filename):
        self.data = []
        self.authors = []
        auths = list(load_data(filename, self.datainfo.dataset, self.datainfo.channels()).items())
        
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
                    self.problems.append(((pi,pj), 1))
                
                    gw = aidx
                    while gw == aidx:
                        gw = random.randint(0, len(self.authors)-1)
                    pbad = random.choice(self.authors[gw])
                    self.problems.append(((pi, pbad), 0))

        random.shuffle(self.problems)

    def __len__(self):
        return int(np.floor(len(self.problems) / self.datainfo.batch_size()))

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
        
        kmin = None #min([len(x) for x in known])
        umin = None #min([len(x) for x in unknown])
        known = sequence.pad_sequences(known, value=0, maxlen=kmin, truncating='post', padding='post')
        unknown = sequence.pad_sequences(unknown, value=0, maxlen=umin, truncating='post', padding='post')
        
        return np.array(known), np.array(unknown)


# streams = 'char', 'words', 'tokens'
def load_data(datafile, dataset="MaCom", channels=('char','word','pos')):
    res = dict()
    path = "data/"+dataset+"/processed/"
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
    for c in channels:
        if c == 'char':
            chres = load_channel(path+datafile+'.csv', fun=util.clean)
        elif c == 'word':
            chres = load_channel(path+datafile+'_words.csv', fun=lambda x: x.split(' '))
        elif c == 'pos':
            chres = load_channel(path+datafile+'_pos.csv', fun=lambda x: x.split(' '))
            
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

def generate_stats(datafile, dataset="MaCom"):
    dinfo = DataInfo(dataset)
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
    cmap = [x for x in cmap if x[1] > dinfo.char_freq_cutoff*ctot]
    cmap.sort(key=lambda x:-x[1])

    wmap = list(wmap.items())
    wmap = [x for x in wmap if x[1] > dinfo.word_freq_cutoff*wtot]
    wmap.sort(key=lambda x:-x[1])
    
    pmap = list(pmap.items())
    pmap.sort(key=lambda x:-x[1])

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

