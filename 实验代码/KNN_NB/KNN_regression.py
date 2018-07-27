import numpy as np
path = 'DATA/regression_dataset/'
# 得到情感表
fileI = open(path + 'train_set.csv')
emot = fileI.readline().strip('\n').split(',')[1:]
fileI.close()
# 得到单词表
voc = set([])
for name in ('train', 'validation', 'test'):
	fileI = open(path + name + '_set.csv')
	fileI.readline()
	for line in fileI:
		voc |= set(line.split(',')[1 if name == 'test' else 0].split())
	fileI.close()

# 得到虚词表，在单词表去除虚词表
fw_set = set([])
fileI = open('formword.txt')
for line in fileI:
	fw_set |= set(line.split())
fileI.close()
voc = voc - fw_set
voc_tuple = tuple(voc)

# 读取训练集，验证集，测试集并初始化
def getSet(fpath, target=False, predict=False):
	my_set = []
	fileI = open(fpath)
	fileI.readline()
	for line in fileI:
		row = line.split(',')
		item = {}
		if target:
			item['emot_Pr'] = list(map(float, row[1:]))
		if predict:
			item['emot_predict_Pr'] = [-1] * 6
		item['word_id'] = []
		for w in (set(row[0 if target else 1].split()) & voc):
			item['word_id'] += [voc_tuple.index(w)]
		my_set += [item]
	fileI.close()
	return my_set

train_set = getSet(path + 'train_set.csv', target=True)
validation_set = getSet(path + 'validation_set.csv', True, True)
test_set = getSet(path + 'test_set.csv', predict=True)

# kNN algorithm
k = 20 # k = 60
epsilon = 0.0001
def calc_distance(a, b):
	# 曼哈顿距离 & 欧式距离下降准确率
	c = [x for x in a if x in b]
	# dist = len(a) + len(b) - 2 * len(set(a) & set (b)) # 曼哈顿
	# dist = (2 * len(c) / (len(a) + len(b)) )
	# dist = len(c) ** 4
	dist = ( len(c)*(len(a)+len(b)) / (2*len(a)*len(b)) )**4  #调和平均数
	return 1 / (epsilon + dist)
	# return dist # 曼哈顿
def find_kNN(test_item, train_set, k=1):
	# 找k近邻，返回k个(下标, 距离)组成的序列
	distance = []
	for i in range(len(train_set)):
		dist = calc_distance(test_item['word_id'], train_set[i]['word_id'])
		distance += [(i, dist)]
	distance = sorted(distance, key = lambda x : x[1])
	return distance[:k]
def calc_emotionP(test_item, train_set, knn_res):
	# 计算情感概率
	k = len(knn_res)
	l_emot = len(emot)
	dist = list(map(lambda x: x[1], knn_res))
	dist = np.array(dist)
	dist_inv = 1 / (epsilon + dist) # 分母非零
	dist_inv /= sum(dist_inv) # 归一化
	# if max(list(dist_inv)) - min(list(dist_inv)) != 0:		
	# 	dist_inv = ( dist_inv - min(list(dist_inv)) ) / ( max(list(dist_inv)) - min(list(dist_inv)) )#标准化
	
	emot_Pr_matrix = []
	for i in range(l_emot):
		emot_Pr_matrix += [list(map(lambda x: train_set[x[0]]['emot_Pr'][i], knn_res))]
	emot_Pr_matrix = np.array(emot_Pr_matrix)
	#进行克罗克内积，将矩阵扩展后再相乘
	emot_Pr_matrix *= np.kron(dist_inv, np.ones([l_emot, 1]))
	res = np.sum(emot_Pr_matrix, axis=1) #按列求和
	res /= sum(res) # 再次归一化，减小浮点误差
	return list(res) 

for item in test_set:
	knn_res = find_kNN(item, train_set, k)
	item['emot_predict_Pr'] = calc_emotionP(item, train_set, knn_res)

# 输出文件
import csv
def getResult(filestring,my_set,k):
	for item in my_set:
		knn_res = find_kNN(item, train_set, k)
		item['emot_predict_Pr'] = calc_emotionP(item, train_set, knn_res)
	fileO = open(filestring, 'w', newline='')
	writer = csv.writer(fileO)
	writer.writerow(['textid'] + emot)
	for i in range(len(my_set)):
		itemres = [str(i+1)] + my_set[i]['emot_predict_Pr']
		writer.writerow(itemres)
	fileO.close()
	return

filestring = "KNN_regression_validation_res.csv"
getResult(filestring,validation_set,k)
filestring = "15352116_Hongziqi_KNN_regression.csv"
getResult(filestring,test_set,k)
