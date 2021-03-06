# preprocess
import re
import sys
import numpy
from polyglot.text import Text
import textstat

PROFILES = {
        "PAN13":{
            "lang":"en",
            "dir":"../data/PAN13",
            "remove_first":False,
            "remove_names":False,
            "txt_max_length": 10000,
            "txt_min_length": 200,
            "txt_max_length_sentence": 500,
            "author_min_num": 2
            },
        "MaCom":{
            "lang":"da",
            "dir":"../data/MaCom",
            "remove_first":True,
            "remove_names":True,
            "txt_max_length": 30000,
            "txt_min_length": 200,
            "txt_max_length_sentence": 500,
            "author_min_num": 5
            }
        }

def clean(txt):
    txt = re.sub(r'\$NL\$', '\n', txt)
    txt = re.sub(r'\$SC\$', ';', txt)
    txt = ''.join([i if ord(i) < 256 else ' ' for i in txt])
    #txt = re.sub(r'\U00100078', '', txt)
    #txt = re.sub(r'\uf020', '', txt)
    return txt.strip()
def unclean(txt):
    txt = re.sub(r'\n', '$NL$', txt)
    txt = re.sub(r';', '$SC$', txt)
    return txt

if len(sys.argv) != 3:
    print('Wrong number of args...')
    print('Usage: python preprocess.py <PROFILE> <DATA_FILE_NAME>')
    sys.exit(1)

profile = sys.argv[1]
if profile not in PROFILES:
    print("Profile missing")
    sys.exit(1)
profile = PROFILES[profile]
dfile = sys.argv[2]

############## CONFIG
text_upper_threshold = profile["txt_max_length"]
text_lower_threshold = profile["txt_min_length"]
text_sentence_threshold = profile["txt_max_length_sentence"] 

author_num_texts_threshold = profile["author_min_num"]

path_raw = profile["dir"]+"/raw/"
path_pro = profile["dir"]+"/processed/"

############# Count file size
print("Starting preprocessing...")
with open(path_raw+dfile+'.csv', 'r', encoding="utf8") as f:
    length = len([i for i,l in enumerate(f)])
    print("Total number of lines in file: "+str(length))

############# Rough filter texts and authors
print("\nRough filtering...")
authors = dict()
percent = 0
with open(path_raw+dfile+'.csv', 'r', encoding="utf8") as f:
    for i,l in enumerate(f):
        if i == 0:
            continue
            
        if float(100*i)/float(length) > percent:
            print(str(percent)+"%")
            percent += 10

        l = l.strip().split(';')
        uid, ts, text = l
        ctext = clean(text)
        
        if len(ctext) == 0:
            continue
        
        if uid not in authors:
            authors[uid] = []
        
        polytext = Text(ctext, hint_language_code=profile["lang"])
        sentences = polytext.sentences
         
        if len(sentences) > text_sentence_threshold:
            continue
        
        if profile["remove_first"]:
            ctext = ctext[200:]
         
        if len(ctext) > text_upper_threshold or len(ctext) < text_lower_threshold:
            continue
            
        authors[uid].append((ts,text,len(sentences)))
    print("100%\n")

print("Removing duplicates")
cdup = 0
cnodup = 0
for k,ls in list(authors.items()):
    newls = []
    flag = set()
    cdup += len(ls)
    cnodup += len(ls)
    for ts,text,nsen in ls:
        if text in flag:
            cnodup -= 1
            continue
        flag.add(text)
        newls.append((ts,text,nsen))

    authors[k] = newls

print("Removed "+str(cdup-cnodup)+" duplicates ("+str(cnodup)+"/"+str(cdup)+")")
print("")

cauth = 0
texts = []
for k,ls in authors.items():
    if len(ls) > author_num_texts_threshold:
        cauth += 1
        for ts,txt,nsen in ls:
            texts.append((k,ts,txt,nsen))

print("Total number of texts: "+str(len(texts))+"/"+str(length))
print("Total number of authors: "+str(cauth)+"/"+str(len(authors)))
authors = None

############# Extract features
print("\nExtracting features...")
length = len(texts)
timestamps = []
ntexts = []
words = []
pos   = []
meta  = []
percent = 0
for i,(uid,ts,text,nsen) in enumerate(texts):
    if float(100*i)/float(len(texts)) > percent:
        percent = int(float(100*i)/float(len(texts)))
        print(str(percent)+"%")
        percent += 1
        
    text = clean(text)
    
    polytext = Text(text, hint_language_code=profile["lang"])
    postags = polytext.pos_tags

    cntmap = dict()
    cntmap['NOUN'] = 0
    cntmap['VERB'] = 0
    for _,pt in postags:
        if pt not in cntmap:
            cntmap[pt] = 0
        cntmap[pt] += 1

    flesch  = textstat.flesch_reading_ease(text)
    smog    = textstat.smog_index(text)
    coleman = textstat.coleman_liau_index(text)
    ari     = textstat.automated_readability_index(text)
    linsear = textstat.linsear_write_formula(text)
    gunfog  = textstat.gunning_fog(text)

    if profile['remove_names']:
        i = 0
        ntext = ''
        for (word, tag) in postags:
            l = len(word)
            while text[i:i+l] != word:
                ntext += text[i]
                i += 1
            if tag == 'PROPN':
                # Remove
                word = '$PROPN$'
            ntext += word
            i += l
        text = ntext
    
    wlist = [(unclean(x[0]) if not profile['remove_names'] or x[1] != 'PROPN' else '$PROPN$') for x in postags]
    plist = [x[1] for x in postags]
    wordmap = dict()
    for (w,p) in postags:
        if p in ('PUNCT', 'PROPN', 'SYM', 'X', 'NUM'):
            continue
        if w not in wordmap:
            wordmap[w] = 0
        wordmap[w] += 1
    
    if profile["remove_first"]:
        text = text[200:]
        wlist = wlist[20:]
        plist = plist[20:]
                    
    text = unclean(text)
    
    timestamps.append((uid,ts))
    ntexts.append((uid,text))
    words.append((uid, " ".join(wlist)))
    pos.append((uid, " ".join(plist)))

    meta.append((nsen, cntmap['NOUN'], cntmap['VERB'], flesch, smog, coleman, ari, linsear, gunfog, wordmap))
print("100%\n")
    
texts = ntexts

print("\nSaving...")
with open(path_pro+dfile+'_ts.csv', 'w', encoding='utf8') as fts,\
        open(path_pro+dfile+'.csv', 'w', encoding='utf8') as ftext,\
        open(path_pro+dfile+'_words.csv', 'w', encoding='utf8') as fword,\
        open(path_pro+dfile+'_pos.csv', 'w', encoding='utf8') as fpos,\
        open(path_pro+dfile+'_meta.csv', 'w', encoding='utf8') as fmeta,\
        open(path_pro+dfile+'_wrdmap.csv', 'w', encoding='utf8') as fwrdmp:
    fmeta.write('uid;sentences;nouns;verbs;flesch;smog;coleman;ari;linesar;gunfog\n')
    for (uid, ts), (_, text), (_, words), (_, pos), dmeta in zip(timestamps, texts, words, pos, meta):
        fts.write(uid+";"+ts+"\n")
        ftext.write(uid+";"+text+"\n")
        fword.write(uid+";"+words+"\n")
        fpos.write(uid+";"+pos+"\n")
        (nsen, nnoun, nverb, flesch, smog, coleman, ari, linsear, gunfog, wordmap) = dmeta
        fmeta.write(uid+';'+str(nsen)+';'+str(nnoun)+';'+str(nverb)+';'+str(flesch)+';'+str(smog)+';'
                +str(coleman)+';'+str(ari)+';'+str(linsear)+';'+str(gunfog)+'\n')
        wlist = ";".join([str(w)+','+str(c) for w,c in sorted(list(wordmap.items()), key=lambda x: x[1])])
        fwrdmp.write(uid+';'+wlist+'\n')
    
