import copy
import math
import random
import numpy as np

def sigmoid(X):
	return 1 / ( 1 + np.exp(-X) )
def sigmoid_inv(X):
	return np.log( X / (1-X) )
def scale(X):
	# return (X - X.mean()) / X.std()
	return (X - X.min())/(X.max()-X.min())

# def groupValidation(train, k=10): #get the validation by separating the train
# 	L = []
# 	m = len(train)
# 	n = math.ceil(m / k)
# 	id_vali = list(range(m))
# 	random.shuffle(id_vali)
# 	L += [list(map(lambda y: id_vali[y] ,map(lambda i: i*n, range(k))))] 
# 	return list(map(operator.add, *L))

class BilayerRModel(object):
	def __init__(self, train_X, train_y, vali_X, vali_y, hwidth=10, times=500):
		m, xdim = train_X.shape
		alpha, eps, lam = 0.000001, 1e-7, 0.7
		nr = np.random.rand
		self.w = [nr(xdim, hwidth), nr(hwidth, 1)]
		self.b = [nr(1, hwidth), nr(1, 1)]
		self.losses = {}
		self.losses['train'] = []
		self.losses['validation'] = []

		wc, bc = copy.deepcopy(self.w), copy.deepcopy(self.b)

		t_mse = self.sse(train_X, train_y) 
		v_mse = self.sse(vali_X, vali_y) 
		self.losses['train'] += [t_mse]
		self.losses['validation'] += [v_mse]
		J0 = t_mse + lam * self.rd()
		# k = math.ceil(m)
		for t in range(times):
			for i in  random.sample(range(m), m):
				o1 = sigmoid(np.dot(train_X[[i]], self.w[0]) + self.b[0]) # (1, hwidth)
				o2 = np.dot(o1, self.w[1]) + self.b[1] # (1, 1)
				g = o2 - train_y[i] # (1, 1)
				e = o1 * (1 - o1) * self.w[1].T * g # (1, hwidth)
				self.w[1] = (1 - alpha*lam) * self.w[1] - alpha * g * o1.T
				self.b[1] = (1 - alpha*lam) * self.b[1] - alpha * g
				self.w[0] = (1 - alpha*lam) * self.w[0] - alpha * np.dot(train_X[[i]].T, e)
				self.b[0] = (1 - alpha*lam) * self.b[0] - alpha * e
			t_mse = self.sse(train_X, train_y) 
			v_mse = self.sse(vali_X, vali_y) 
			self.losses['train'] += [t_mse]
			self.losses['validation'] += [v_mse]
			J = t_mse + lam * self.rd()
			print('times:', t, ' mse:', t_mse,' J:', J, ' J0:', J0)
			if J < J0:
				wc, bc = copy.deepcopy(self.w), copy.deepcopy(self.b)
				if (J0 - J) / J < eps:
					print('times:', t)
					break
				J0 = J
			else:
				# if np.linalg.norm(self.w[0] - w0[0]) < epsilon * np.linalg.norm(w0[0]):
				# 	print('times:', t)
				# 	break
				# alpha /= 2
				self.w, self.b = wc, bc
		pass
	def loss(self):
		return self.losses
	def predict(self, test_X):
		m = test_X.shape[0]
		b = sigmoid(np.dot(test_X, self.w[0]) + self.b[0])  # (m, width)
		p = np.dot(b, self.w[1]) + self.b[1]       # (m, 1)
		return p
	def sse(self, train_X, train_y):
		p = self.predict(train_X)
		diff = p - train_y
		return (diff * diff).sum()/train_X.shape[0]

	def rd(self):
		w0, w1, b0, b1 = self.w[0], self.w[1], self.b[0], self.b[1]
		return (w0*w0).sum() + (w1*w1).sum() + (b0*b0).sum() + (b1*b1).sum()

class Batch_BilayerRModel(object):
	def __init__(self, train_X, train_y, vali_X, vali_y, hwidth=10, times=500):
		m, xdim = train_X.shape
		alpha, eps, lam = 0.01, 1e-7, 0.8
		nr = np.random.rand
		# nr = np.ones(())
		self.w = [nr(xdim, hwidth), nr(hwidth, 1)]
		self.b = [nr(1, hwidth), nr(1, 1)]
		wc, bc = copy.deepcopy(self.w), copy.deepcopy(self.b)	
		self.losses = {}
		self.losses['train'] = []
		self.losses['validation'] = []
		t_mse = self.sse(train_X, train_y) 
		v_mse = self.sse(vali_X, vali_y) 
		self.losses['train'] += [t_mse]
		self.losses['validation'] += [v_mse]
		J0 = t_mse + lam * self.rd()
		# k = math.ceil(m)
		k = m
		for t in range(times):
			for i in range(m):
			# for i in random.sample(range(m), k):	
				o1 = sigmoid(np.dot(train_X[[i]], wc[0]) + bc[0]) # (1, hwidth)
				o2 = np.dot(o1, wc[1]) + bc[1]
				g = o2 - train_y[i] # (1, 1)
				e = o1 * (1 - o1) * wc[1].T * g # (1, hwidth)
				self.w[1] -= alpha * g * o1.T / k
				self.b[1] -= alpha * g / k
				self.w[0] -= alpha * np.dot(train_X[[i]].T, e) / k
				self.b[0] -= alpha * e / k
			self.w[1] -= alpha*lam * self.w[1]
			self.b[1] -= alpha*lam * self.b[1]
			self.w[0] -= alpha*lam * self.w[0]
			self.b[0] -= alpha*lam * self.b[0]
			t_mse = self.sse(train_X, train_y) 
			v_mse = self.sse(vali_X, vali_y) 
			self.losses['train'] += [t_mse]
			self.losses['validation'] += [v_mse] 
			J = t_mse + lam * self.rd()
			print('times:', t, ' mse:', t_mse,' J:', J, ' J0:', J0)
			if J < J0:
				wc, bc = copy.deepcopy(self.w), copy.deepcopy(self.b)
				if (J0 - J) / J < eps:
					print('times:', t)
					break
				J0 = J
			else:
				self.w, self.b = wc, bc
		pass
	def loss(self):
		return self.losses
	def predict(self, test_X):
		m = test_X.shape[0]
		b = sigmoid(np.dot(test_X, self.w[0]) + self.b[0])  # (m, width)
		p = np.dot(b, self.w[1]) + self.b[1]       # (m, 1)
		return p
	def sse(self, train_X, train_y):
		p = self.predict(train_X)
		diff = p - train_y
		return (diff * diff).sum() / len(train_X)
	def rd(self):
		w0, w1, b0, b1 = self.w[0], self.w[1], self.b[0], self.b[1]
		return (w0*w0).sum() + (w1*w1).sum() + (b0*b0).sum() + (b1*b1).sum()

class Mini_Batch_BilayerRModel(object):
	def __init__(self, train_X, train_y, vali_X, vali_y, hwidth=14, times=100):
		m, xdim = train_X.shape
		alpha, eps, lam = 1e-5, 1e-10, 0.1
		nr = np.random.rand
		self.w = [nr(xdim, hwidth), nr(hwidth, 1)]
		self.b = [nr(1, hwidth), nr(1, 1)]
		self.losses = {}
		self.losses['train'] = []
		self.losses['validation'] = []
		wc, bc = copy.deepcopy(self.w), copy.deepcopy(self.b)
		t_mse = self.sse(train_X, train_y) 
		v_mse = self.sse(vali_X, vali_y) 
		self.losses['train'] += [t_mse]
		self.losses['validation'] += [v_mse]
		J0 = t_mse + lam * self.rd()
		k = math.ceil(0.577 * m)
		for t in range(times):
			for i in random.sample(range(m), k):		
				o1 = sigmoid(np.dot(train_X[[i]], self.w[0]) + self.b[0]) # (1, hwidth)
				o2 = np.dot(o1, self.w[1]) + self.b[1] # (1, 1)
				g = o2 - train_y[i] # (1, 1)
				e = o1 * (1 - o1) * self.w[1].T * g # (1, hwidth)
				self.w[1] = (1 - alpha*lam) * self.w[1] - alpha * g * o1.T
				self.b[1] = (1 - alpha*lam) * self.b[1] - alpha * g
				self.w[0] = (1 - alpha*lam) * self.w[0] - alpha * np.dot(train_X[[i]].T, e)
				self.b[0] = (1 - alpha*lam) * self.b[0] - alpha * e		
			t_mse = self.sse(train_X, train_y) 
			v_mse = self.sse(vali_X, vali_y) 
			self.losses['train'] += [t_mse]
			self.losses['validation'] += [v_mse] 
			J = t_mse + lam * self.rd()
			print('times:', t, ' mse:', t_mse,' J:', J, ' J0:', J0)
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
				# alpha /= 2
				self.w, self.b = wc, bc
		pass
	def loss(self):
		return self.losses
	def predict(self, test_X):
		m = test_X.shape[0]
		b = sigmoid(np.dot(test_X, self.w[0]) + self.b[0])  # (m, width)
		p = np.dot(b, self.w[1]) + self.b[1]       # (m, 1)
		return p
	def sse(self, train_X, train_y):
		p = self.predict(train_X)
		diff = p - train_y
		return (diff * diff).sum() / len(train_X)
	def rd(self):
		w0, w1, b0, b1 = self.w[0], self.w[1], self.b[0], self.b[1]
		return (w0*w0).sum() + (w1*w1).sum() + (b0*b0).sum() + (b1*b1).sum()


class Little(object):
	def __init__(self, train_X, train_y, hwidth=1, times=1):
		m, xdim = train_X.shape
		alpha, lam =1,  0
		nr = np.random.rand
		self.w = [nr(xdim, hwidth), nr(hwidth, 1)]
		self.b = [nr(1, hwidth), nr(1, 1)]
		print('initialize')
		print('w[0]:', self.w[0].T, 'b[0]:', self.b[0])
		print('w[1]:', self.w[1], 'b[1]:', self.b[1])
		for t in range(times):
			for i in  range(m):
				print('Forwark Pass:')
				o1 = sigmoid(np.dot(train_X[[i]], self.w[0]) + self.b[0]) # (1, hwidth)
				o2 = np.dot(o1, self.w[1]) + self.b[1] # (1, 1)
				print('hiddenlayer:', o1,'outlayer', o2)
				g = o2 - train_y[i] # (1, 1)
				# print(train_y[i])
				e = o1 * (1 - o1) * self.w[1].T * g # (1, hwidth)
				# print('g: '+str(g) + '\n')
				# print('e:' +str(e) + '\n')
				self.w[1] = (1 - alpha*lam) * self.w[1] - alpha * g * o1.T
				self.b[1] = (1 - alpha*lam) * self.b[1] - alpha * g
				self.w[0] = (1 - alpha*lam) * self.w[0] - alpha * np.dot(train_X[[i]].T, e)
				self.b[0] = (1 - alpha*lam) * self.b[0] - alpha * e
				print('Backwark Pass:')
				print('w[0]:', self.w[0].T, 'b[0]:', self.b[0])
				print('w[1]:', self.w[1], 'b[1]:', self.b[1])
		pass	

