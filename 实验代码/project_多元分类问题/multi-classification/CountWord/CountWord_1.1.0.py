"""
version 1.1.0
测试集句子加权平均，权重为标准差

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
        item = {'data': cleanString(l[1].lstrip()).split('<sssss>')}
        dataset.append(item)
    file.close()
    return label, dataset

# In [3]:
print('loading data ...', end='', flush=True)
label, train_set = readData('MulLabelTrain.ss')
_, test_set = readData('MulLabelTest.ss')
print('done')

# In [4]:
print('counting words ...')
word_statistics = {'LOW': Counter(), 'MID': Counter(), 'HIG': Counter()}
progress = 0
for l, item in zip(label, train_set):
    for sentnc in item['data']:
        word_statistics[l] += Counter(sentnc.split())
    progress += 1
    if progress % 100 == 0:
    	print('\b'*10, round(progress/62522*100,2), '% ', end='', flush=True)
print('\ndone')

# In [5]:
print('generating matrix ...', end='', flush=True)
LMH = {}
sum_L = sum(word_statistics['LOW'].values())
sum_M = sum(word_statistics['MID'].values())
sum_H = sum(word_statistics['HIG'].values())
vocabulary = map(lambda x:set(word_statistics[x].keys()), ['LOW', 'MID', 'HIG'])
vocabulary = reduce(operator.or_,vocabulary)
for voc in vocabulary:
    count = [word_statistics['LOW'][voc]/sum_L, word_statistics['MID'][voc]/sum_M, word_statistics['HIG'][voc]/sum_H]
    count = np.array(count)
    sigma = count.sum()
    count = count / sigma
    LMH[voc] = {'count': count}
print('done')

# In [6]:
print('predicting ...', end='', flush=True)
res = []
for item in test_set:
    pred, weight = [], []
    for sentns in item['data']:
        sigma = np.zeros(3)
        for wd in sentns.split():
            try:
                sigma += LMH[wd]['count']
            except:
                pass
        if sigma.sum():
            sigma = sigma / sigma.sum()
        else:
            sigma = np.ones(3) / 3
        pred.append(sigma)
        weight.append(sigma.std())
    pred = np.array(pred)
    weight = np.array([weight])
    if weight.sum() == 0:
        pred = pred.mean(axis=0)
    else:
        weight = weight / weight.sum()
        pred = np.dot(weight, pred).reshape(3) # (1,n) x (n,3)
    res.append(pred)
print('done')

# In [7]:
whatlabel = ('LOW', 'MID', 'HIG')
res_ = np.array(res).argmax(axis=1)
res_ = list(map(lambda x:whatlabel[x], res_))

print('making submission ...', flush=True)

# In [8]:
file = open('CountWord_submission_1.1.0.csv', 'w')
for r in res_:
    file.write(r + '\n')
file.close()

# In [9]:
file = open('CountWord_softprob_1.1.0.csv', 'w')
for r in res:
    out = ','.join(map(str,r))
    file.write(out + '\n')
file.close()

print('finish')
