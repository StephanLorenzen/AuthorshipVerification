import random


CHAR_MAP_SIZE = 100
WORD_MAP_SIZE = 10000


# streams = 'char', 'words', 'tokens'
def load_data(datafile):    
    res = dict()
    with open(datafile+".csv", 'r') as ftxt:
        for l in ftxt:
            l = l.strip().split(";")
            uid = l[0]
            txt = l[1]
            if uid not in res:
                res[uid] = [[], [], []]
            res[uid][0].append(txt)

    with open(datafile+"_words.csv", 'r') as fwrd:
        for l in fwrd:
            l = l.strip().split(";")
            uid = l[0]
            words = l[1].split(' ')
            res[uid][2].append(words)
        
    with open(datafile+"_pos.csv", 'r') as fpos:
        for l in fpos:
            l = l.strip().split(";")
            uid = l[0]
            pos = l[1].split(' ')
            res[uid][2].append(pos)
    
    return res

def generate_stats(datafile):
    data = load_data(datafile)
    text = ""
    words = []
    for _,d in data.items():
        for txt in d[0]
        texts += txt
        for wls in d[1]:
            words += wls
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
    cmap.sort(key=x:-x[0])
    cmap = cmap[:CHAR_MAP_SIZE]

    wmap = list(wmap.items())
    wmap.sort(key=x:-x[0])
    wmap = wmap[:WORD_MAP_SIZE]

    with open('processed_data/cmap.txt', 'w') as f:
        for c in cmap:
            f.write(str(c[0])+";"+str(c[1])+"\n")
    with open('processed_data/wmap.txt', 'w') as f:
        for w in wmap:
            f.write(str(w[0])+";"+str(w[1])+"\n")

def load_stats():
    cmap, wmap = dict(), dict()
    with open('processed_data/cmap.txt', 'r') as f:
        for i,l in enumerate(f):
            l = l.strip().split(";")
            cmap[l[0].strip()] = i+1
    with open('processed_data/wmap.txt', 'r') as f:
        for i,l in enumerate(f):
            l = l.strip().split(";")
            wmap[l[0].strip()] = i+1
    return cmap, wmap

def prepare_text(txt, wrds, pos, cmap, wmap):
    chars = []
    words = []
    for c in txt:
        chars.append(cmap.get(c, 0))
    while len(chars) < 30000:
        chars.append(0)
    for w in wrds:
        words.append(wmap.get(w, 0))
    words = words[:10000]
    while len(words) < 10000:
        words.append(0)
    return chars, words, []



def get_siamese_set(datafile):
    authors = list(load_data(datafile).items())
    authors_processed = []

    cmap, wmap = load_stats()

    for (uid, data) in authors:
        texts = data[0]
        words = data[1]
        pos   = data[3]
    
        texts_processed = []
        words_processed = []
        pos_processed   = []
        for txt, wrds, tags in zip(texts,words,pos):
            (txtp, wrdsp, tagsp) = prepare_text(txt, wrds, tags, cmap, wmap)
            text_processed.append(txtp)
            words_processed.append(wrdsp)
            pos_processed.append(tagsp)

        authors_processed.append((uid, [texts_processed, words_processed, pos_processed]))

    authors = None

    dataset = []
    for aidx (author, data) in enumerate(authors_processed):
        for i in range(len(data)):
            for j in i+1,range(len(data)):
                dataset.append((data[i], data[j], 1))
        
                gw = aidx
                while gw == aidx:
                    gw = random.randint(0, len(authors_processed)-1)
                godtxt = random.choice(data)
                badtxt = random.choice(authors_processed[gw][1])
                dataset.append((godtxt, badtxt, 0))

    dataset = random.shuffle(dataset)

    return [(x[0], x[1]) for x in dataset], [x[2] for x in dataset]
    

