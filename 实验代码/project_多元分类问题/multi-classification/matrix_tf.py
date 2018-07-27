import numpy as np
import copy
import random
import operator
import re
import math
from functools import reduce
# import MC_MODEL as mc
# import matplotlib.pyplot as plt
from collections import Counter
# import seaborn as sns
# % matplotlib inline
# sns.set()
stopwords = [
	"i",          "me",         "my",         "myself",     "we",        
	"our",        "ours",       "ourselves",  "you",        "your",      
	"yours",      "yourself",   "yourselves", "he",         "him",       
	"his",        "himself",    "she",        "her",        "hers",      
	"herself",    "it",         "its",        "itself",     "they",      
	"them",       "their",      "theirs",     "themselves", "what",      
	"which",      "who",        "whom",       "this",       "that",      
	"these",      "those",      "am",         "is",         "are",       
	"was",        "were",       "be",         "been",       "being",     
	"have",       "has",        "had",        "having",     "do",        
	"does",       "did",        "doing",      "would",      "should",    
	"could",      "ought",      "i'm",        "you're",     "he's",      
	"she's",      "it's",       "we're",      "they're",    "i've",      
	"you've",     "we've",      "they've",    "i'd",        "you'd",     
	"he'd",       "she'd",      "we'd",       "they'd",     "i'll",      
	"you'll",     "he'll",      "she'll",     "we'll",      "they'll",   
	"isn't",      "aren't",     "wasn't",     "weren't",    "hasn't",    
	"haven't",    "hadn't",     "doesn't",    "don't",      "didn't",    
	"won't",      "wouldn't",   "shan't",     "shouldn't",  "can't",     
	"cannot",     "couldn't",   "mustn't",    "let's",      "that's",    
	"who's",      "what's",     "here's",     "there's",    "when's",    
	"where's",    "why's",      "how's",      "a",          "an",        
	"the",        "and",        "but",        "if",         "or",        
	"because",    "as",         "until",      "while",      "of",        
	"at",         "by",         "for",        "with",       "about",     
	"against",    "between",    "into",       "through",    "during",    
	"before",     "after",      "above",      "below",      "to",        
	"from",       "up",         "down",       "in",         "out",       
	"on",         "off",        "over",       "under",      "again",     
	"further",    "then",       "once",       "here",       "there",     
	"when",       "where",      "why",        "how",        "all",       
	"any",        "both",       "each",       "few",        "more",      
	"most",       "other",      "some",       "such",       "no",        
	"nor",        "not",        "only",       "own",        "same",      
	"so",         "than",       "too",        "very",       "will"
]
def single(string):
	li = list(string)
	i =[0]+list(filter(lambda x:li[x]!=li[x-1],range(1,len(li))))
	return ''.join([li[x] for x in i])	
def clearstring(string):
	string = re.sub('<sssss>' , ' ', string)
	string = re.sub('- *[lr]rb *-', ' ', string)
	string = re.sub('(^| )http:[^ ]+', ' ', string)
	string = re.sub('(^| )[A-Za-z]( |$)', ' ', string)
	string = re.sub('(^| )\'[^ ]+', ' ', string)
	string = re.sub('(^| )n\'t ', ' ', string)
	string = re.sub('[^A-Za-z !?/]+', '', string)
	string = re.sub('(^| )[A-Za-z]( |$)', ' ', string)
	string = re.sub('/', ' ', string)
	string = string.split()
	# string = set(string) - set(stopwords)
	_str = []
	for x in string:
		if len(x) > 7:
			x = single(x)
		if x not in stopwords: 
			_str += [x]
# 	_str = [x for x in string if x not in stopwords]
	string = ' '.join(_str)
	return string

def readDataSet(fpath):
	label = []
	Dataset = []
	labelvoc = {'LOW':[], 'MID':[], 'HIG':[], '?':[]}
	file = open(fpath, encoding='UTF-8')
	for line in file:
		l = line.split('\t\t')
		label.append(l[0])
		words = clearstring(l[1].strip('\n')).split()
		labelvoc[l[0]] += words
		Dataset.append(words)
	file.close()
	return label, Dataset, labelvoc
def writefile(filename, res):
    file = open(filename,'w')
    for i in res:
        file.write(i+'\n')
    file.close()
    return
def samplebalance(label, train_x, length):
    l = list(map(lambda x:x[1] ,filter(lambda x: x[0]=='LOW',zip(label,range(len(label))))))
    m = list(map(lambda x:x[1] ,filter(lambda x: x[0]=='MID',zip(label,range(len(label))))))
    h = list(map(lambda x:x[1] ,filter(lambda x: x[0]=='HIG',zip(label,range(len(label))))))
    index = l[:length] + m[:length] + h[:length] 
    label_ = list(map(lambda x: label[x],index))
    train_ = np.array(list(map(lambda x: train_x[x],index)))
    return label_, train_

label, train_set, label_voc = readDataSet('MulLabelTrain.ss')
_, test_set, test_voc = readDataSet('MulLabelTest.ss')
vlabel, valid_set, _ = readDataSet('SmallValid.ss')

label_voc['LOW'] = Counter(label_voc['LOW'])
label_voc['MID'] = Counter(label_voc['MID'])
label_voc['HIG'] = Counter(label_voc['HIG'])
test_voc['?'] = Counter(test_voc['?'])
same_word = set(label_voc['LOW']& label_voc['MID'] & label_voc['HIG'])
voc_couter = label_voc['LOW'] + label_voc['MID'] + label_voc['HIG'] + test_voc['?']
lfword = []
useless = []
lf = 3
threshold =0.0001
w = voc_couter.items()
for key, value in w:
	if value < lf:
		lfword += [key]
for wd in same_word:
	l = label_voc['LOW'][wd]
	m = label_voc['MID'][wd]
	h = label_voc['HIG'][wd]
	if np.array([l,m,h]).var() < threshold:
		useless += [wd] 
count = np.array(list(voc_couter.values()))
lenght = len(count[count>100])
voc_id = tuple(map(lambda x: x[0], voc_couter.most_common(lenght))) 

def tfset(dataset, voc_id):
    myset = []
    vc = set(voc_id)
    for line in dataset:
        item = [0]*lenght
        cont = Counter(line)
        sum_ = sum(cont.values())
        for i in set(cont.keys()) & vc:
            item[voc_id.index(i)] = cont[i] / sum_
        myset.append(item)
    return myset

train_tf = tfset(train_set, voc_id)
test_tf = tfset(test_set, voc_id)
train_tf = np.array(train_tf)
test_tf = np.array(test_tf)
valid_tf = np.array(tfset(valid_set, voc_id))

def writefile(filename, res):
    file = open(filename,'w')
    for i in res:
        file.write(i+'\n')
    file.close()
    return
def labelTonum(label):
    L_y = list(map(lambda x:1 if x =='LOW' else 0, label))
    L_y = np.array([L_y]).T
    M_y = list(map(lambda x:1 if x =='MID' else 0, label))
    M_y = np.array([M_y]).T
    H_y = list(map(lambda x:1 if x =='HIG' else 0, label))
    H_y = np.array([H_y]).T
    return L_y, M_y, H_y
def numTolabel(y):
    name = ['LOW','MID','HIG']
    return list(map(lambda x:name[x],y))
def samplebalance(label, train_x, length, start=0):
    l = list(map(lambda x:x[1] ,filter(lambda x: x[0]=='LOW',zip(label,range(len(label))))))
    m = list(map(lambda x:x[1] ,filter(lambda x: x[0]=='MID',zip(label,range(len(label))))))
    h = list(map(lambda x:x[1] ,filter(lambda x: x[0]=='HIG',zip(label,range(len(label))))))
    index = l[start:length] + m[start:length] + h[start:length] 
    label_ = list(map(lambda x: label[x],index))
    train_ = np.array(list(map(lambda x: train_x[x],index)))
    return label_, train_
def check_label(res ,std):
    print(len(res))
    print(len(std))
    return (np.array(res) == np.array(std)).sum() / len(std)
def dist_Train_test(train_X, label, test_item):
	dist = train_X - test_item
	dist = np.sum(dist * dist, axis=1)
	item = zip(dist, label)
	return sorted(item, key=lambda x:x[0], reverse=True)

def evaluate(TP, FN, TN, FP):
	Accuracy = (TP + TN) / np.float64(TP + FP + TN + FN)
	return Accuracy
def check(res, std):
	count = [0]*4
	for r, s in zip(res, std):
		wrong = int(r != s)
		neg = int(s != 1)
		count[neg * 2 + wrong] += 1
	print(evaluate(*count))
	return 
def sigmoid(X):
	return 1 / (1 + np.exp(-X))
class LogicalRegression(object):
	"""docstring for LogicalRegression"""
	def __init__(self, train_X, train_y, weights=None, times=1000, alpha=0.01, c=1, style='batch'):
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
			check(res, vali_y)
		return
    
class Mini_Batch_BilayerRModel(object):
	def __init__(self, train_X, label, hwidth=14, times=10000, owidth=3):
		m, xdim = train_X.shape
		alpha, eps = 1e-5, 1e-10
		nr = np.random.rand
		self.w = [nr(xdim, hwidth), nr(hwidth, owidth)]
		self.b = [nr(1, hwidth), nr(1, owidth)]
# 		self.losses = {'train':[], 'validation':[]}
		wc, bc = copy.deepcopy(self.w), copy.deepcopy(self.b)
		train_y = self.matrix_train(label) # (m, 3)
		J0 = self.logloss(train_X, train_y)
# 		self.check(train_X, label)
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
slabel, strain_tf = samplebalance(label, train_tf, 17000)
####################################KNN Validtion_tf##########################
K = [1,2,3,4,5,6,7,8,9,10,11,20,30,70,100,120,222]
# K = [1]
resK = []
_pred = []
for k in K:
    KNN_pred = []
    for item in valid_tf:
        NN = dist_Train_test(strain_tf, slabel, item)
        KNN_pred += [Counter(list(map(lambda x:x[1],NN[:k]))).most_common(1)[0][0]]
    ac = check_label(KNN_pred, svlabel)
    if ac > 0.4:
        _pred.append(KNN_pred)
        resK.append(k)
print(resK)
####################################KNN Validtion_tf##########################