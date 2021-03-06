"""
version 3.3.0
stop word
soft clean train set & iter = 2


简单统计关于类别的词频，得到词频矩阵。
矩阵行数为单词数，列为LOW, MID, HIG
列归一化后再行归一化
得到的矩阵的行向量即为每个单词关于每个类别似然概率
"""

# In [ ]:
import operator
import re

import numpy as np

from functools import reduce
from collections import Counter

# In [ ]:
labels = ('LOW', 'MID', 'HIG')
label2int = {'LOW':0, 'MID':1, 'HIG':2}
stopword = [
    "i",          "me",         "my",         "myself",     "we", "us",       
    "our",        "ours",       "ourselves",  "you",        "your",      
    "yours",      "yourself",   "yourselves", "he",         "him",       
    "his",        "himself",    "she",        "her",        "hers",      
    "herself",    "it",         "its",        "itself",     "they",      
    "them",       "their",      "theirs",     "themselves", "what",      
    "which",      "who",        "whom",       "this",       "that",      
    "these",      "those",      "am",         "is",         "are",       
    "was",        "were",       "be",         "been",       "being",     
    "have",       "has",        "had",        "having",     "do",        
    "does",       "did",        "doing",      "thats",      "the",        
    "and",        "whos",       "whats",      "heres",      "theres",    
    "whens",      "wheres",     "whys",       "hows",       "a",         
    "an",         "when",       "where",      "why",        "with",
    "how",        "own"
]

def cleanString(string):
    string = re.sub('- *[lr]rb *-', ' ', string)
    string = re.sub('(^| )http:[^ ]+', ' ', string)
    string = re.sub('/', ' ', string)
    string = re.sub('[^A-Za-z !?<>]+', '', string)
    string = re.sub('(^| )[A-Za-z]( |$)', ' ', string)
    return string
def readData(fpath):
    label = []
    dataset = [] 
    file = open(fpath, encoding='UTF-8')
    for line in file:
        l = line.split('\t\t')
        label.append(l[0])
        item = cleanString(l[1].strip()).split('<sssss>')
        dataset.append(item)
    file.close()
    return label, dataset
def generating_matrix(stat):
    LMH = {}
    sum_L = sum(stat['LOW'].values())
    sum_M = sum(stat['MID'].values())
    sum_H = sum(stat['HIG'].values())
    vocabulary = map(lambda x:set(stat[x].keys()), labels)
    vocabulary = reduce(operator.or_,vocabulary)
    for voc in vocabulary:
        count = [stat['LOW'][voc]/sum_L, stat['MID'][voc]/sum_M, stat['HIG'][voc]/sum_H]
        count = np.array(count)
        sigma = count.sum()
        count = count / sigma
        LMH[voc] = {'count': count}
    return LMH

# In [ ]:
print('loading data ...', end='', flush=True)
train_label, train_set = readData('MulLabelTrain.ss')
_, test_set = readData('MulLabelTest.ss')
print('done')

# In [ ]:
def dropStopwords(dataset):
    for i in range(len(dataset)):
        for j in range(len(dataset[i])):
            wd = dataset[i][j].split()
            wd = list(filter(lambda x: x not in stopword, wd))
            dataset[i][j] = ' '.join(wd)
    return dataset

# In [ ]:
print('drop stop words ...', end='', flush=True)
train_set = dropStopwords(train_set)
test_set = dropStopwords(test_set)
print('done')

# In [ ]:
print('counting words ...')
print('init iteration', flush=True)
word_statistics = {'LOW': Counter(), 'MID': Counter(), 'HIG': Counter()}
progress = 0
for l, item in zip(train_label, train_set):
    for sentnc in item:
        word_statistics[l] += Counter(sentnc.split())
    progress += 1
    if progress % 100 == 0:
        print('\b'*10, round(progress/62522*100,2), '% ', end='', flush=True)
max_iteration = 2
watchlist = (
    'perfect', 'love', 'amazing', 'delicious',
    'wait', 'again', 'will', 'nt', 'very',
    'nice', 'bad', 'decent', 'just', 'however', 'but',
    'ok', 'better', 'great', '!', '?'
)
for t in range(max_iteration):
    LMH = generating_matrix(word_statistics)
    for wl in watchlist:
        print(wl+':', LMH[wl])
    if t == max_iteration - 1:
        break
    print('\niteration', t+1, flush=True)
    word_statistics = {'LOW': Counter(), 'MID': Counter(), 'HIG': Counter()}
    progress = 0
    for l, item in zip(train_label, train_set):
        for sentnc in item:
            sigma = np.zeros(3)
            for wd in sentnc.split():
                try:
                    sigma += LMH[wd]['count']
                except:
                    pass
            if sigma[label2int[l]] > 0.15:
                word_statistics[l] += Counter(sentnc.split())
        progress += 1
        if progress % 100 == 0:
            print('\b'*10, round(progress/62522*100,2), '% ', end='', flush=True)

print('\ndone')

# In [ ]:
print('generating matrix ...', end='', flush=True)
LMH = generating_matrix(word_statistics)
print('done')

# In [ ]:
print('predicting ...', end='', flush=True)
res = []
for item in test_set:
    pred = []
    for sentnc in item:
        sigma = np.zeros(3)
        for wd in sentnc.split():
            try:
                sigma += LMH[wd]['count']
            except:
                pass
        if sigma.sum():
            sigma = sigma / sigma.sum()
        else:
            sigma = np.ones(3) / 3
        pred.append(sigma)
    pred = np.array(pred).mean(axis=0)
    res.append(pred)
print('done')

# In [ ]:
res_ = np.array(res).argmax(axis=1)
res_ = list(map(lambda x:labels[x], res_))

print('making submission ...')

# In [ ]:
file = open('CountWord_submission_3.0.1.csv', 'w')
for r in res_:
    file.write(r + '\n')
file.close()

# In [ ]:
file = open('CountWord_softprob_3.0.1.csv', 'w')
for r in res:
    out = ','.join(map(str,r))
    file.write(out + '\n')
file.close()

print('finish')
