# preprocess

import sys
import numpy
from polyglot.text import Text

if len(sys.argv) < 2:
    print('No arg given...')
    sys.exit(1)

path = 'D:/DABAI/Data/MagnusOgAugust/data/before_preprocessing/'
ppath = 'preprocessed_data/'

dfile = sys.argv[1]

############## CONFIG
text_upper_threshold = 30000
text_lower_threshold = 200
text_sentence_threshold = 500

author_num_texts_threshold = 5

############# Count file size
print("Starting preprocessing...")
with open(path+dfile+'.csv', 'r', encoding="utf8") as f:
    length = len([i for i,l in enumerate(f)])
    print("Total number of lines in file: "+str(length))

############# Rough filter texts and authors
print("\nRough filtering...")
authors = dict()
percent = 0
with open(path+dfile+'.csv', 'r', encoding="utf8") as f:
    for i,l in enumerate(f):
        if i == 0:
            continue
            
        if float(100*i)/float(length) > percent:
            print(str(percent)+"%")
            percent += 10

        l = l.strip().split(';')
        uid = l[0]
        text = l[2]
        
        if uid not in authors:
            authors[uid] = []
        
        polytext = Text(text, hint_language_code='da')
        sentences = polytext.sentences
        
        if len(sentences) > text_sentence_threshold:
            continue
        
        text     = text[len(polytext.sentences[0]):]
        
        if len(text) > text_upper_threshold or len(text) < text_lower_threshold:
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
words = []
pos   = []
percent = 0
for i,(uid,text) in enumerate(texts):
        
    if float(100*i)/float(len(texts)) > percent:
        print(str(percent)+"%")
        percent += 1    
    
    text = text.replace('$NL$', ' ')
    
    polytext = Text(text, hint_language_code='da')
    postags = polytext.pos_tags

    words.append((uid, " ".join([x[0] for x in postags])))
    pos.append((uid, " ".join([x[1] for x in postags])))
print("100%\n")
    
print("\nSaving...")
with open(ppath+dfile+'.csv', 'w', encoding='utf8') as ftext, open(ppath+dfile+'_words.csv', 'w', encoding='utf8') as fword, open(ppath+dfile+'_pos.csv', 'w', encoding='utf8') as fpos:
    for (uid, text), (_, words), (_, pos) in zip(texts, words, pos):
        ftext.write(uid+";"+text+"\n")
        fword.write(uid+";"+words+"\n")
        fpos.write(uid+";"+pos+"\n")
    
    