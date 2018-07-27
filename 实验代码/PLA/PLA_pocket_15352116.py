import common_method as cm
import numpy as np
import csv

fpath = 'Lab3/'
train = cm.getData(fpath+'train.csv', target=True) 
val = cm.getData(fpath+'val.csv', target=True)
test = cm.getData(fpath+'test.csv', predict=True)

def pocketPLA(test_x,w,times):
	pocket = [-1,w]         #利用F1参数进行评测指标，对F1进行初始化
	for i in range(times):  #设置迭代次数
		for item in test_x:
			if cm.sign(sum(item['x']*w)) !=  item['y']:
				w += item['y']*np.array(item['x'])   #与原始一样，预测不同就更新w
		(Ac,Re,Pre,F1) = cm.check(test_x,w) #遍历完样本后将w与之前的w进行比较
		if pocket[0] < F1:   #如果得到的比较好，就将w放进口袋，同时更新F1指标
			pocket = [F1,w]
	# print(w)
	# print('Time : ' + str(i) + '    mypocket_F1 :' + str(pocket[0]))
	return pocket			#返回口袋

w0 = np.ones(len(train[0]['x']))
t_pocket = pocketPLA(train,w0,500)
(Ac,Re,Pre,F1) = cm.check(val,t_pocket[1])

