# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os
import gc
import numpy as np
gc.enable()


# %%
dataPrefix = '../data/'
dirr = list(os.walk(dataPrefix))[0][2]
words = []
encodingSet = set(["dvd.task.train", "MR.task.test", "MR.task.train"])
for i in dirr:
    if i.split('.')[-1] == 'unlabel':
        continue
    try:
        for j in open(dataPrefix+i,
                            encoding="utf-8" if i not in encodingSet else "ISO-8859-1").readlines():
            words.extend(j.split('\t')[1].split())
    except IndexError:
        print(i)
        for j in open(dataPrefix+i,
                            encoding="utf-8" if i not in encodingSet else "ISO-8859-1").readlines():
            words.extend(j.split())
words = set(words)


# %%
vect_words = set()
with open('../glove.twitter.27B.200d.txt', 'r',encoding='utf-8') as f:
    for line in f:
        vect_words.add(line.split(' ')[0])

words = set(words).intersection(set(vect_words))
# words = list(words)


# %%
vocab_size = len(words)
vocab_t = {w: idx for idx, w in enumerate(words)}
open('../vocab_t','w',encoding='utf-8').write(str(vocab_t))
open('../words','w',encoding='utf-8').write(str(words))


# %%
# vocab_t = eval(open('../vocab_t',encoding='utf-8').readline())
# words = eval(open('../words',encoding='utf-8').readline())


# %%
vectors = {}
with open('../glove.twitter.27B.200d.txt', 'r',encoding='utf-8') as f:
    for line in f:
        vals = line.rstrip().split(' ')
        if vals[0] in words:
            vectors[vals[0]] = [float(x) for x in vals[1:]]
open('../vectors','w',encoding='utf-8').write(str(vectors))


# %%
# vectors = eval(open('../vectors',encoding='utf-8').readline())


# %%
words = list(words)
vocab_size = len(words)
vocab_dim = len(vectors[words[0]])
vector_t = np.zeros((vocab_size, vocab_dim))
for word, v in vectors.items():
    if word == '<unk>':
        continue
    vector_t[vocab_t[word], :] = v
np.save('../vector_t',vector_t)

null.tpl [markdown]
# vector_t = np.fromfile('../vector_t.npy')
