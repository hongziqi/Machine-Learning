import numpy as np
import math
import pandas as pd
import collections
import csv
from functools import reduce
path = 'DATA/regression_dataset/'
emotion = ('anger', 'disgust', 'fear', 'joy', 'sad', 'surprise')

# 得到虚词表
fw_set = set([])
fileI = open('formword.txt')
for line in fileI:
	fw_set |= set(line.split())
fileI.close()

# 得到训练集，验证集，测试集并且初始化
def getSet(fpath, target=False, predict=False):
	voc = set()
	my_set = []
	fileI = open(fpath)
	fileI.readline()
	for line in fileI:
		col = line.strip('\n').split(',')
		item = {}
		if target:
			item['emot'] = list(map(float, col[1:7]))
		if predict:
			item['pred_emot'] = [-1] * 6
		item['words'] = [x for x in col[0 if target else 1].split() if not (x in fw_set)]   #删去虚词
		# item['words'] = col[0 if target else 1].split() #保留虚词

		item['count'] = collections.Counter(item['words'])
		voc |= set(col[0 if target else 1].split())
		my_set += [item]
	fileI.close()
	return (voc, my_set)

(train_voc, train_set) = getSet(path + 'train_set.csv', target=True)
(validation_voc, validation_set) = getSet(path + 'validation_set.csv', True, True)
Len_voc = len(train_voc - fw_set) #5-》0.33
# Len_voc = len(train_voc) #5 -》0.33  ,1->0.27  20->0.26
# Len_voc = len(train_voc) #5->0.29
# Len_voc = len(train_voc | validation_voc) #5->0.29
test_set = getSet(path + 'test_set.csv', predict=True)[1]
# print(validation_set)

lambda1 = 5

def f(wd, train):
	num = (train['count'][wd] + lambda1)
	denom = (len(train['words']) + lambda1*Len_voc)
	return num / denom
	# return train['count'][wd] / len(train['words'])
lambda2 = 0.0001
def getPredit_emot(test_set,train_set):
	for item in test_set:
		prob = []
		for e in range(len(emotion)): 
			sigma = 0
			for i_train in train_set: #在某一类别下遍历训练集，根据公式计算测试样本在这一类别下发生的概率
				total_words = len(i_train['words'])
				p = [f(wd, i_train) for wd in item['words']]
				sigma += i_train['emot'][e]*reduce(lambda a,b:a*b, p)
			prob += [sigma]
		item['pred_emot'] = list(np.array(prob) / sum(prob)) #归一化
		# item['pred_emot'] = list(np.array(prob) / (lambda2 + sum(prob))) #归一化
	return
getPredit_emot(validation_set, train_set)
getPredit_emot(test_set, train_set)

def getResult(filestring,my_set):
	fileO = open(filestring, 'w', newline='')
	writer = csv.writer(fileO)
	writer.writerow(['textid'] + list(emotion))
	for i in range(len(my_set)):
		itemres = [str(i+1)] + my_set[i]['pred_emot']
		writer.writerow(itemres)
	fileO.close()
	return
filestring = 'NB_regression_validation_res.csv'
getResult(filestring,validation_set)
filestring = '15352116_Hongziqi_NB_regression.csv'
getResult(filestring,test_set)
