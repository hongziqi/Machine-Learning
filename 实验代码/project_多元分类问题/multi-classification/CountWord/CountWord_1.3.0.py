"""
version 1.3.0
drop noise sentences in train set

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

# In [3]:
# labels = ('MID', 'LOW', 'HIG') # M 放在 L 前面，是为了在M和L概率相等时，通过argmax优先选择MID
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
        item = cleanString(l[1].lstrip()).split('<sssss>')
        # item = {'data': cleanString(l[1].lstrip()).split('<sssss>')}
        # item['enable'] = [True] * len(item['data'])
        # item['inlabel'] = [''] * len(item['data'])
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

# In [4]:
print('loading data ...', end='', flush=True)
train_label, train_set = readData('MulLabelTrain.ss')
_, test_set = readData('MulLabelTest.ss')
print('done')

# In [5]:
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
for t in range(max_iteration):
    LMH = generating_matrix(word_statistics)
    print('love:', LMH['love'])
    if t == max_iteration - 1:
        break
    print('\niteration', t, flush=True)
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
            if labels[sigma.argmax()] == l:
                word_statistics[l] += Counter(sentnc.split())
        progress += 1
        if progress % 100 == 0:
            print('\b'*10, round(progress/62522*100,2), '% ', end='', flush=True)

print('\ndone')

# In [6]:
print('generating matrix ...', end='', flush=True)
LMH = generating_matrix(word_statistics)
print('done')

# # In [7]:
# print('predicting ...', end='', flush=True)
# res = []
# for item in test_set:
#     pred = []
#     for sentnc in item:
#         sigma = np.zeros(3)
#         for wd in sentnc.split():
#             try:
#                 sigma += LMH[wd]['count']
#             except:
#                 pass
#         if sigma.sum():
#             sigma = sigma / sigma.sum()
#         else:
#             sigma = np.ones(3) / 3
#         pred.append(sigma)
#     pred = np.array(pred).mean(axis=0)
#     res.append(pred)
# print('done')

# # In [8]:
# res_ = np.array(res).argmax(axis=1)
# res_ = list(map(lambda x:labels[x], res_))

# print('making submission ...')

# # In [9:
# file = open('CountWord_submission_1.3.0.csv', 'w')
# for r in res_:
#     file.write(r + '\n')
# file.close()

# # In [10]:
# file = open('CountWord_softprob_1.3.0.csv', 'w')
# for r in res:
#     out = ','.join(map(str,r))
#     file.write(out + '\n')
# file.close()

# print('finish')
# ########################################################
# In [6]:
print('predicting ...', end='', flush=True)
res = []
for item in test_set:
    pred, weight = [], []
    for sentns in item:
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
file = open('CountWord_submission_1.3.1.csv', 'w')
for r in res_:
    file.write(r + '\n')
file.close()

# In [9]:
file = open('CountWord_softprob_1.3.1.csv', 'w')
for r in res:
    out = ','.join(map(str,r))
    file.write(out + '\n')
file.close()

print('finish')
