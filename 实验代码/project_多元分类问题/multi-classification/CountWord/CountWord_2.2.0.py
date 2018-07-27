"""
version 2.0.0
log p 

简单统计关于类别的词频，得到词频矩阵。
矩阵行数为单词数，列为LOW, MID, HIG
列归一化后再行归一化
得到的矩阵的行向量即为每个单词关于每个类别似然概率
"""

# In [1]:
import operator
import re

import numpy as np

from functools import reduce
from collections import Counter

# In [2]:
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
    "an",        "to",          "from",       "up",         "down",       
    "in",         "out",        "on",         "off",        "over",       
    "under",      "again",      "here",       "there" 
]
# In [3]:
labels = ('LOW', 'MID', 'HIG')
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

# In [4]:
print('loading data ...', end='', flush=True)
train_label, train_set = readData('MulLabelTrain.ss')
_, test_set = readData('MulLabelTest.ss')
print('done')

# In [5]:
def dropStopwords(dataset):
    for i in range(len(dataset)):
        for j in range(len(dataset[i])):
            wd = dataset[i][j].split()
            wd = list(filter(lambda x: x not in stopword, wd))
            dataset[i][j] = ' '.join(wd)
    return dataset

# In [6]:
print('drop stop words ...', end='')
train_set = dropStopwords(train_set)
test_set = dropStopwords(test_set)
print('done')

# In [7]:
print('counting words ...')
word_statistics = {'LOW': Counter(), 'MID': Counter(), 'HIG': Counter()}
progress = 0
for l, item in zip(train_label, train_set):
    for sentns in item:
        word_statistics[l] += Counter(sentns.split())
    progress += 1
    if progress % 100 == 0:
        print('\b'*10, round(progress/62522*100,2), '% ', end='', flush=True)
print('\ndone')

# In [8]:
print('generating matrix ...', end='', flush=True)
LMH = {}
sum_L = sum(word_statistics['LOW'].values())
sum_M = sum(word_statistics['MID'].values())
sum_H = sum(word_statistics['HIG'].values())
vocabulary = map(lambda x:set(word_statistics[x].keys()), labels)
vocabulary = reduce(operator.or_,vocabulary)
for voc in vocabulary:
    count = [word_statistics['LOW'][voc]/sum_L, word_statistics['MID'][voc]/sum_M, word_statistics['HIG'][voc]/sum_H]
    count = np.array(count)
    sigma = count.sum()
    count = count / sigma
    LMH[voc] = {'count': count}
print('done')

# In [9]:
print('predicting ...', end='', flush=True)
res = []
for item in test_set:
    pred = []
    for sentns in item:
        sigma = np.zeros(3)
        for wd in sentns.split():
            try:
                sigma += np.log(LMH[wd]['count']+1e-16)
            except:
                pass
        a = np.exp(sigma[1]-sigma[2])
        b = np.exp(sigma[2]-sigma[0])
        c = np.exp(sigma[0]-sigma[1])
        x = 1/(1+b+1/c)
        y = 1/(1+1/a+c)
        z = 1/(1+a+1/b)
        pred.append(np.array([x,y,z]))
    pred = np.array(pred).mean(axis=0)
    res.append(pred)
print('done')

# In [10]:
res_ = np.array(res).argmax(axis=1)
res_ = list(map(lambda x:labels[x], res_))

print('making submission ...')

# In [11]:
file = open('CountWord_submission_2.2.0.csv', 'w')
for r in res_:
    file.write(r + '\n')
file.close()

# In [12]:
file = open('CountWord_softprob_2.2.0.csv', 'w')
for r in res:
    out = ','.join(map(str,r))
    file.write(out + '\n')
file.close()

print('finish')
