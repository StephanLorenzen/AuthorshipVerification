# preprocess
import re
import sys
import numpy
from polyglot.text import Text

PROFILES = {
        "PAN13":{
            "lang":"en",
            "dir":"PAN13",
            "remove_first":False,
            "remove_names":False,
            "txt_max_length": 10000,
            "txt_min_length": 200,
            "txt_max_length_sentence": 500,
            "author_min_num": 2
            },
        "MaCom":{
            "lang":"da",
            "dir":"MaCom",
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
    return txt
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
        uid = l[0]
        text = l[2]
        ctext = clean(text)
        
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
            
        authors[uid].append(text)
    print("100%\n")
        
cauth = 0
texts = []
for k,ls in authors.items():
    if len(ls) > author_num_texts_threshold:
        cauth += 1
        for t in ls:
            texts.append((k,t))

print("Total number of texts: "+str(len(texts))+"/"+str(length))
print("Total number of authors: "+str(cauth)+"/"+str(len(authors)))
authors = None

############# Extract features
print("\nExtracting features...")
length = len(texts)
ntexts = []
words = []
pos   = []
percent = 0
for i,(uid,text) in enumerate(texts):
    if float(100*i)/float(len(texts)) > percent:
        percent = int(float(100*i)/float(len(texts)))
        print(str(percent)+"%")
        percent += 1
    
    text = clean(text)
    
    polytext = Text(text, hint_language_code=profile["lang"])
    postags = polytext.pos_tags
    
    if profile['remove_names']:
        i = 0
        ntext = ''
        for (word, tag) in postags:
            l = len(word)
            #import pdb; pdb.set_trace()
            while text[i:i+l] != word:
                ntext += text[i]
                i += 1
            if tag == 'PROPN':
                # Remove
                word = '$PROPN$'
            ntext += word
            #pdb.set_trace()
            i += l
        text = ntext
    
    wlist = [(x[0] if x[1] != 'PROPN' else '$PROPN$') for x in postags]
    plist = [x[1] for x in postags]
    if profile["remove_first"]:
        text = text[200:]
        wlist = wlist[20:]
        plist = plist[20:]
                    
    text = unclean(text)
    
    ntexts.append((uid,text))
    words.append((uid, " ".join(wlist)))
    pos.append((uid, " ".join(plist)))
print("100%\n")
    
texts = ntexts

print("\nSaving...")
with open(path_pro+dfile+'.csv', 'w', encoding='utf8') as ftext, open(path_pro+dfile+'_words.csv', 'w', encoding='utf8') as fword, open(path_pro+dfile+'_pos.csv', 'w', encoding='utf8') as fpos:
    for (uid, text), (_, words), (_, pos) in zip(texts, words, pos):
        ftext.write(uid+";"+text+"\n")
        fword.write(uid+";"+words+"\n")
        fpos.write(uid+";"+pos+"\n")
    
    
