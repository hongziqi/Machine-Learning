import numpy as np
import random
import math
import operator
import copy
from functools import reduce


def sigmoid(X):
	return 1 / (1 + np.exp(-X))
class LogicalRegression(object):
	"""docstring for LogicalRegression"""
	def __init__(self, train_X, train_y, weights=None, times=1000000, alpha=0.0001, c=2, style='batch'):
		self.train_X = train_X
		self.train_y = train_y
		self.times = times
		self.alpha = alpha
		self.c = c
		gradient = []
		if weights == None:
			weights = np.zeros((1,train_X.shape[1])) # 1*M
		# alpha = 0.000001
		epsilon = 1e-6
		self.w0 = weights.T # M*1
		p = sigmoid(np.dot(train_X, self.w0)) # N*M X M*1 = N*1
		J0 = self.logistic_cost(p, train_y)
		K0 = K00 = J0
		for t in range(times):
			p_y = p - train_y
			w = self.w0 - alpha * np.dot(train_X.T, p_y) 
			# print(np.dot(train_X.transpose(), p_y))
			gradient.append(np.sum(-np.dot(train_X.T, p_y)))
			# w = self.w0 - alpha * np.dot(train_X.transpose(), p_y) / len(p) - lambda_ * self.w0 / len(p) #40*1
			p = sigmoid(np.dot(train_X, w))
			J = self.logistic_cost(p, train_y)
			if t > 7 and K00 == J:
				break
			K00 = K0
			K0 = J
			if J < J0: # not convergent
				J0 = J
				self.w0 = w
			else:
				alpha /= c
				if np.linalg.norm(w - self.w0) < epsilon * np.linalg.norm(self.w0):
					# print('times: ', t)
					break
		return # M*1
	def LR(self, train_X, train_y,weights=None):
		gradient = []
		if weights == None:
			weights = np.zeros((1,train_X.shape[1])) # 1*M
		# alpha = 0.000001
		epsilon = 1e-3
		w0 = weights.T # M*1
		# print(w0)
		p = sigmoid(np.dot(train_X, w0)) # N*M X M*1 = N*1
		J0 = self.logistic_cost(p, train_y)
		K0 = K00 = J0
		for t in range(self.times):
			p_y = p - train_y
			w = w0 - self.alpha * np.dot(train_X.T, p_y) 
			# print(np.dot(train_X.transpose(), p_y))
			gradient.append(np.sum(-np.dot(train_X.T, p_y)))
			# w = w0 - alpha * np.dot(train_X.transpose(), p_y) / len(p) - lambda_ * w0 / len(p) #40*1
			p = sigmoid(np.dot(train_X, w))
			J = self.logistic_cost(p, train_y)
			if t > 7 and K00 == J:
				break
			K00 = K0
			K0 = J
			if J < J0: # not convergent
				J0 = J
				w0 = w
			else:
				self.alpha /= self.c
				if np.linalg.norm(w - w0) < epsilon * np.linalg.norm(w0):
					# print('times: ', t)
					break
		return w0 # M*1
	def logistic_cost(self, p, train_y):
		_p = copy.deepcopy(p)
		_p[train_y == 0] = 1 - _p[train_y == 0]
		return -sum(np.log(_p+0.000001)) 
	def predict(self, test_X):
		return  sigmoid(np.dot(test_X, self.w0))
	def groupValidation(self, k=10):
		n = self.train_X.shape[0]
		m = math.ceil(n/k)
		n_id = list(range(n))
		random.shuffle(n_id)
		L = []
		L += list(map(lambda x:n_id[x:x+m] ,map(lambda i:i*m, range(10))))
		return L
	def evaluate(self, TP, FN, TN, FP):
		Accuracy = (TP + TN) / np.float64(TP + FP + TN + FN)
		return Accuracy
	def check(self, res, std):
		count = [0]*4
		for r, s in zip(res, std):
			wrong = int(r != s)
			neg = int(s != 1)
			count[neg * 2 + wrong] += 1
		print(self.evaluate(*count))
		return 
	def crossValidation(self):
		L = self.groupValidation()
		for i in range(len(L)):
			cp_L = copy.deepcopy(L)
			vali_id = cp_L.pop(i)
			vali_X, vali_y = self.train_X[vali_id], self.train_y[vali_id]
			tra_x, tra_y = self.train_X[reduce(operator.add,cp_L)], self.train_y[reduce(operator.add,cp_L)]
			weights = self.LR(tra_x, tra_y)
			res = np.dot(vali_X, weights)
			res = list(map(lambda x: 1 if x > 0 else 0, res))
			self.check(res, vali_y)
		return
# class KNN(object):
# 	def __init__(self, train_X, test_X, k):
# 		for item in test_X:

# 			pass
# 		return
class Mini_Batch_BilayerRModel(object):
	def __init__(self, train_X, label, hwidth=14, times=10000, owidth=3):
		m, xdim = train_X.shape
		alpha, eps = 1e-5, 1e-10
		nr = np.random.rand
		self.w = [nr(xdim, hwidth), nr(hwidth, owidth)]
		self.b = [nr(1, hwidth), nr(1, owidth)]
		# self.losses = {'train':[], 'validation':[]}
		wc, bc = copy.deepcopy(self.w), copy.deepcopy(self.b)
		train_y = self.matrix_train(label) # (m, 3)
		J0 = self.logloss(train_X, train_y)
		#self.check(train_X, label)
		# k = math.ceil(0.777 * m)
		for t in range(times):
			for i in random.sample(range(m), m):		
				o1 = sigmoid(np.dot(train_X[[i]], self.w[0]) + self.b[0]) # (1, hwidth)
				o2 = sigmoid(np.dot(o1, self.w[1]) + self.b[1]) # (1, 3)
				tmp = np.zeros((1,3))
				neg = train_y[i].reshape(o2.shape) == 0
				tmp[neg] = 1/(1-o2[neg])
				tmp[~neg] = -1/o2[~neg]
				g = tmp*o2*(1-o2)  # (1, 3)
				e = o1 * (1 - o1) * np.dot(g, self.w[1].T) # (1, hwidth)
				self.w[1] -= alpha * g * o1.T # (hwidth, 3)
				self.b[1] -= alpha * g
				self.w[0] -= alpha * np.dot(train_X[[i]].T, e) #(50, hwidth)
				self.b[0] -= alpha * e		
			J = self.logloss(train_X, train_y)
			print('times:', t, ' J:', J, ' J0:', J0)
			if t == 99:
				self.check(train_X, label)
			if J < J0:
				wc, bc = copy.deepcopy(self.w), copy.deepcopy(self.b)
				if (J0 - J) / J < eps:
					print('times:', t)
					break
				J0 = J
			else:
				# if np.linalg.norm(self.w[0] - wc[0]) < eps * np.linalg.norm(wc[0]):
				# 	print('times:', t)
				# 	break
				alpha /= 2
				self.w, self.b = wc, bc
		pass
	def loss(self):
		return self.losses
	def predict(self, test_X):
		m = test_X.shape[0]
		b = sigmoid(np.dot(test_X, self.w[0]) + self.b[0])  # (m, hwidth)
		p = sigmoid(np.dot(b, self.w[1]) + self.b[1])       # (m, 3)
		return p
	def matrix_train(self, train_y):
		l = np.array([list(map(lambda x: 1 if x == 'LOW' else 0, train_y))])
		m = np.array([list(map(lambda x: 1 if x == 'MID' else 0, train_y))])
		h = np.array([list(map(lambda x: 1 if x == 'HIG' else 0, train_y))])
		return np.concatenate((l,m,h),axis=0).T # (m, 3)
	def logloss(self, train_X, train_y):
		p = self.predict(train_X)
		tmp = np.zeros(p.shape)
		neg = train_y == 0
		tmp[neg] = np.log(1-p[neg])
		tmp[~neg] = np.log(p[~neg])
		return -sum(sum(tmp))
		# return -sum(sum(train_y*np.log(p) + (1-train_y)*np.log(1-p)))
	def check(self, train_X, label):
		p = self.predict(train_X)
		_res = p.argmax(axis = 1)
		L_name = ['LOW', 'MID', 'HIG']
		res = list(map(lambda x: L_name[x], _res))
		print(res)
		ac = (np.array(res) == np.array(label)).sum() / len(label)
		print(ac)
		return

