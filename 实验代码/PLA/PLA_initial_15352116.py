import common_method as cm
import numpy as np
import csv

#fpath = 'littledata/'
fpath = 'Lab3/'
fname = 'thur78train.csv'
# train = cm.getData(fpath+'train.csv', target=True)
# val = cm.getData(fpath+'val.csv', target=True)
# test = cm.getData(fpath+'test.csv', predict=True)
train = cm.getData(fpath+'thur78train.csv', target=True)
test = cm.getData(fpath+'thur78test.csv', predict=True)
train_X = np.array(list(map(lambda x:x['x'], train))) #m*dim
train_y = np.array([list(map(lambda x:x['y'],train))]).transpose() #m*1
train_y[train_y == -1] = 0
# print(train_X.shape, train_y.shape)
m, dim = train_X.shape  #4000, 66

#***************************原始PLA算法*************************
def initialPLA(test,w=np.ones(dim),times=500):
	for i in range(times):   #迭代次数
		c = m;
		for item in test:
			if cm.sign(sum(item['x']*w)) !=  item['y']:  
				w += item['y']*np.array(item['x'])  #找到预测错误的样本就更新w
			else:
				c -= 1
		if c == 0:   
			print("Time:"+str(times))
			break
	return w #返回最后的w
weights = initialPLA(train,times=1000);
print(weights)
# print("************* This is the result of initial PLA ***************")
print("train:")
cm.check(train, weights)
# print("validation:")
# cm.check(val, weights)
filestring = '15352116_hongziqi_PLA.csv'
cm.getResult(test,weights,filestring)

#**************************逻辑回归算法*************************
def sigmoid(X):
	return 1 / (1 + np.exp(-X))
def logistic_cost(p, train_y, k):
	p[train_y == 0] = 1 - p[train_y == 0] #取出y = 0的数组
	# print(p)
	klogp = np.log(p)
	klogp[train_y == 0] *= k[0]
	klogp[train_y == 1] *= k[1]
	return -sum(klogp)
def logistic_regression(train_X, train_y, weights=None, times=1000):
	if weights == None:
		weights = [0.0] * train_X.shape[1]
	alpha = 1
	epsilon = 1e-10
	kp = train_y[train_y == 1].shape[0] / train_y.shape[0]
	kn = 1 - kp
	k = (kp, kn)
	# k = (0.5, 0.5) #一般情况下设置为等价代价
	w0 = np.array([weights]).transpose() # dim x 1
	p = sigmoid(np.dot(train_X, w0)) # m x 1
	J0 = logistic_cost(p[:], train_y, k)
	for t in range(times):
		# w = w0
		# for j in range(train_X.shape[1]):
		# 	x_j_T = train_X[:, j:j+1].transpose()
		# 	# 写成train_X[:, j]会变成一维向量
		# 	w[j] = w0[j] - alpha * np.dot(x_j_T, p - train_y)
		p_y = p - train_y
		p_y[train_y == 0] *= k[0]
		p_y[train_y == 1] *= k[1] #由于模型的负值类过多，所以设置一个权重
		w = w0 - alpha * np.dot(train_X.transpose(), p_y) #每一个w的更新和整个训练集有关
		p = sigmoid(np.dot(train_X, w))
		J = logistic_cost(p, train_y, k)
		if J < J0: #没有收敛，就继续循环，并且更新损失函数以及更新之后的w
			J0 = J
			w0 = w
		else:	   #损失变大了，可能越过了极小值，用原来的w，步长减半再次计算损失函数
			alpha /= 2
			 #如果更新后的w与上一个w相差不大，就证明收敛了，退出循环
			if np.linalg.norm(w - w0) < epsilon * np.linalg.norm(w0):
				print('times:', t)
				break
	weights = w0.transpose().tolist()[0] #转换成list
	return weights

# weights = cm.read_weights('weights.txt')
# weights = logistic_regression(train_X, train_y)
# print(weights)
# print("************* This is the result of regression algorithm ***************")
# print("train:")
# cm.check(train, weights)
# print("validation:")
# cm.check(val, weights)