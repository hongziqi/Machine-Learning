import numpy as np
import math
import pandas as pd
import collections
import csv
from functools import reduce
path = 'DATA/classification_dataset/'
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
			item['emot'] = col[1]
		if predict:
			item['pred_emot'] = ''
		item['words'] = col[0 if target else 1].split()
		# item['words'] = [x for x in col[0 if target else 1].split() if not (x in fw_set)]
		voc |= set(col[0 if target else 1].split())
		my_set += [item]
	fileI.close()
	return (voc,my_set)

(train_voc, train_set) = getSet(path + 'train_set.csv', target=True)
(validation_voc, validation_set) = getSet(path + 'validation_set.csv', True, True)
(test_voc,test_set )= getSet(path + 'test_set.csv', predict=True)
# sigma = len(train_voc - fw_set)
sigma = len(train_voc | validation_voc - fw_set)
# sigma = len(train_voc | validation_voc)
# print(validation_set)

N = len(train_set)	# 样本总数 
emot_num = collections.Counter(list(map(lambda x: x['emot'], train_set))) # 每种情感的样本数的集合
emotion = tuple(sorted(emot_num.keys()))	# 情感列表，用于索引
denominator = N+len(emotion)
pei = list(map(lambda x: (emot_num[x])/N, emotion))	# 每种情感占样本总数的概率的集合

df = pd.DataFrame(train_set)
words_emot = list(df['words'].groupby(df['emot'])) #根据情感对train_set进行分组
words_emot = list(map(lambda w: reduce(lambda x,y:x+y, list(w[1])), words_emot)) #整合同类别的单词
words_emot = list(map(lambda x: collections.Counter(x), words_emot)) #记录了每个单词出现的频数
words_scnt_emot = list(map(lambda x: len(x), words_emot)) #每种情感下单词的总数（不重复的）
words_cnt_emot = list(map(lambda x: sum(x.values()), words_emot)) #每种情感下单词出现的总数（重复的）

lambda2 = 0.5
def f(x, e):
	#拉普拉斯平滑：
	return math.log((words_emot[e][x]+lambda2)/(words_cnt_emot[e]+sigma*lambda2)) 
	# return (words_emot[e][x]+lambda2)/(words_cnt_emot[e]+sigma*lambda2)
def getPredit_emot(test_set, train_set, validation=False):
	error = 0
	for item in test_set:
		words = tuple(item['words'])
		emot_id = range(len(emotion)) #索引的情感id
		# prob = list(map(lambda e: reduce(lambda x,y: x*y, [f(w,e) for w in words]),emot_id)) 
		#将公式转化为对数求和的方式计算
		prob = list(map(lambda e:math.exp(reduce(lambda x,y:x+y,[f(w,e) for w in words])),emot_id)) 
		P_res = tuple(np.array(pei) * np.array(prob))
		item['pred_emot'] = emotion[P_res.index(max(P_res))] #取得最大概率下的情感类别作为预测结果
		# print(item['pred_emot'])
		if validation:
			if item['pred_emot'] != item['emot']:
				error += 1
			# print(item['pred_emot'] + '=======' + item['emot'])
	return [test_set, error]
(validation_set, error) = getPredit_emot(validation_set, train_set, True)
print(1-error/len(validation_set))
test_set = getPredit_emot(test_set, train_set)[0]
# 输出文件
def getResult(filestring,my_set):
	fileO = open(filestring,'w',newline='')
	writer = csv.writer(fileO)
	writer.writerow(['textid','label'])
	for i in range(len(my_set)):
		itemres = [str(i+1), my_set[i]['pred_emot']]
		writer.writerow(itemres)
	fileO.close()

filestring = 'NB_classification_validation_res.csv'
getResult(filestring,validation_set)
filestring = '15352116_Hongziqi_NB_classification.csv'
getResult(filestring,test_set)


