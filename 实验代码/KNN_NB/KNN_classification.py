import numpy as np 
import csv
path = 'DATA/classification_dataset/' 
# path = 'little_Dataset/' 
# 得到单词表 和 情感表
voc = set([])
emot = set([])
for name in ('train','validation','test'): 
# for name in ('classification_train','test'):
	fileI = open(path + name + '_set.csv')
	fileI.readline()
	for line in fileI:
		# if name == 'classification_train':
		if name == 'train':
			emot |= set([line.strip('\n').split(',')[1]])
		voc |= set(line.split(',')[1 if name == 'test' else 0].split()) 
		# voc |= set(line.split(',')[0].split()) 
	fileI.close()

# 得到虚词表，在单词表去除虚词表
fw_set = set([])
fileI = open('formword.txt')
for line in fileI:
	fw_set |= set(line.split())
fileI.close()
voc = voc - fw_set

voc_tuple = tuple(voc)   #训练集、验证集、测试集出现的单词表（不重复）
emot_tuple = tuple(emot) #情感种类表
# 读取训练集，验证集，测试集并初始化
def getSet(fpath, target=False, predict=False):
	my_set = []
	fileI = open(fpath)
	fileI.readline() #去掉第一行
	for line in fileI: #第二行开始一行一行处理
		row = line.strip('\n').split(',')
		item = {}
		if target:
			item['emot_id'] = emot_tuple.index(row[1])
		if predict:
			item['emot_predict_id'] = -1
		item['word_id'] = []
		#如果在单词表里面就映射到word_id下 ，并且对每一行的单词进行除重
		for w in (set(row[0 if target else 1].split()) & voc): 
			item['word_id'] += [voc_tuple.index(w)]
		my_set += [item]
	fileI.close()
	return my_set
train_set = getSet(path + 'train_set.csv', target=True)
validation_set = getSet(path + 'validation_set.csv', True, True) 
test_set = getSet(path + 'test_set.csv', predict=True)
#kNN algorithm
k = 20
epsilon = 0.0001
def calc_distance(a, b):
	# 曼哈顿距离 & 欧式距离下降准确率
	c = [x for x in a if x in b]
	# dist = len(a) + len(b) - 2 * len(c) # 曼哈顿
	# return dist
	# dist = (2 * len(c) / (len(a) + len(b)))
	# dist = len(c) ** 4
	dist = ( len(c)*(len(a)+len(b)) / (2*len(a)*len(b)) )**4 #调和平均数
	return 1 / (epsilon + dist)
	
	# return dist # 曼哈顿
def find_kNN(test_item, train_set, k=1):
	# 找k近邻，返回k个(下标, 距离)组成的序列
	distance = []
	for i in range(len(train_set)):
		dist = calc_distance(test_item['word_id'],train_set[i]['word_id'])
		distance += [(i, dist)]
	distance = sorted(distance, key = lambda x : x[1])
	return distance[:k]
def emot_Vector(emot_id):
	vect = np.zeros(len(emot_tuple))
	vect[emot_id] = 1
	return vect
# # 用于观测数据(调参用)
# def getSentence(wordsId):
# 	text = ''
# 	for w in wordsId:
# 		text += voc_tuple[w] + ' '
# 	return text
def getPredict_id(test_set, train_set, validation=False):
	error = 0
	for item in test_set:
		knn_res = find_kNN(item, train_set, k)#前K个在训练集的下标以及与测试样例之间的距离 
		emot_res = np.zeros(len(emot_tuple)) #初始化结果(0,0,0,0,0,0,0)
		for i in range(k):
			#将前K个情感转换为数组进行相加
			emot_res += emot_Vector(train_set[knn_res[i][0]]['emot_id']) 
		pred_id = list(emot_res).index(max(list(emot_res)))	# 找到最大值的下标,即为预测情感下标
		item['emot_predict_id'] = pred_id #更新测试文本中的预测情感
		if validation : 
			if item['emot_predict_id'] != item['emot_id']: #在验证集中,如果预测情感与标准情感不同则error+1
				error += 1
	return [test_set, error] 

(validation_set, validation_error)= getPredict_id(validation_set,train_set,True) 
print(1 - validation_error/len(validation_set)) 
test_set = getPredict_id(test_set, train_set)[0]
# 输出文件
def getResult(filestring,my_set):
	fileO = open(filestring,'w',newline='')
	writer = csv.writer(fileO)
	writer.writerow(['textid','label'])
	for i in range(len(my_set)):
		itemres = [str(i+1), emot_tuple[my_set[i]['emot_predict_id']]]
		writer.writerow(itemres)
	fileO.close()

filestring = 'KNN_classification_validation_res.csv'
getResult(filestring,validation_set)
filestring = '15352116_Hongziqi_KNN_classification.csv'
getResult(filestring,test_set)
# k = 10
# for item in validation_set:
# 	knn_res = find_kNN(item, train_set, k)
# 	emot_res = np.array([0.0]*6)
# 	print('******************K******************')
# 	for ki in range(k):
# 		print('distance :' + str(knn_res[ki][1]) + '  ' + 'index: ' + str(knn_res[ki][0]) + ' ' + getSentence(train_set[knn_res[ki][0]]['word_id']) + ' ' + emot_tuple[train_set[knn_res[ki][0]]['emot_id']])
# 	
