import os
import random

# %%
dataPrefix = './data_orginal/'
datarefix = './data/'
dirr = [i for i in list(os.walk(dataPrefix))[0][2]
        if i.split('.')[-1] == 'train']
encodingSet = set(["dvd.task.train", "MR.task.test", "MR.task.train"])



# %%
for i in dirr:
    train = open(
        dataPrefix+i, encoding="utf-8" if i not in encodingSet else "ISO-8859-1").readlines()
    random.shuffle(train)
    tname = i.split('.')[0] + '.task.train'
    vname = i.split('.')[0] + '.task.valid'
    open(datarefix+tname,'w+', encoding="utf-8" if i not in encodingSet else "ISO-8859-1").writelines(
        train[:int(0.8*len(train))])
    open(datarefix+vname,'w+', encoding="utf-8" if i not in encodingSet else "ISO-8859-1").writelines(
        train[int(0.8*len(train)):])
