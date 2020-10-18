import os
import gc
import re
import random
import numpy as np
gc.enable()


d = ['books','dvd','electronics','kitchen']
dataPrefix = 'data_orginal/'
datarefix = 'data/'
sentences = []
prop = '0'
regex = '(<review_text>)([\s\S]*?)(</review_text>)'
for i in d:
    f = open(dataPrefix + 'temp/' + i + '.task', 'w', encoding='utf-8')
    for j in  list(os.walk(dataPrefix + i))[0][2]:
        if j.split('.')[0] == 'negative':
            prop = '0'
        elif j.split('.')[0] == 'positive':
            prop = '1'
        else:
            continue
        h = re.findall(regex, open(dataPrefix + i + '/' + j, encoding='utf-8').read())
        sentences = [prop + '\t' + i[1].replace('\n',' ').replace('"',' ').replace('  ',' ') + '\n' for i in h]
        for k in sentences:
            f.write(k)
    f.close()

for i in list(os.walk(dataPrefix + 'temp/'))[0][2]:
    with open(dataPrefix + 'temp/' + i, 'r', encoding='utf-8') as f:
        reviews = f.readlines()
    for j in range(10):
        random.shuffle(reviews)
    with open(datarefix + i + '.train', 'w', encoding='utf-8') as f:
        for k in reviews[:1600]:
            f.write(k)
    with open(datarefix + i + '.test', 'w', encoding='utf-8') as f:
        for k in reviews[1600:]:
            f.write(k)
    with open(datarefix + i + '.valid', 'w', encoding='utf-8') as f:
        for k in reviews[1600:]:
            f.write(k)


words = []
for i in list(os.walk(datarefix))[0][2]:
    with open(datarefix + i, encoding="utf-8") as f:
        words.extend(f.read().split())
words = set(words)


vect_words = set()
with open('glove.twitter.27B.200d.txt', 'r',encoding='utf-8') as f:
    for line in f:
        vect_words.add(line.split(' ')[0])

words = words.intersection(vect_words)
vocab_size = len(words)
vocab_t = {w: idx for idx, w in enumerate(words)}
open('vocab_t','w',encoding='utf-8').write(str(vocab_t))


vectors = {}
with open('glove.twitter.27B.200d.txt', 'r',encoding='utf-8') as f:
    for line in f:
        vals = line.rstrip().split(' ')
        if vals[0] in words:
            vectors[vals[0]] = [float(x) for x in vals[1:]]
words = list(words)
vocab_size = len(words)
vocab_dim = len(vectors[words[0]])
vector_t = np.zeros((vocab_size, vocab_dim))
for word, v in vectors.items():
    if word == '<unk>':
        continue
    vector_t[vocab_t[word], :] = v
np.save('vector_t',vector_t)

