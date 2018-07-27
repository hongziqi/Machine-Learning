import numpy as np
import pandas as pd
import math
import random
import operator
import copy
from functools import reduce
from collections import Counter
import pylab as pl

global_var = {}
def entropy(samples, attr='y'):
	gb = samples.groupby(attr).groups.values() #group by 'y' initially
	p = np.array(list(map(len, gb)))
	p = p / sum(p)
	i = (p > 1e-6)
	p[i] = p[i] * np.log2(p[i]) # calculate the entropy when p != 0
	return -sum(p)
def id3(samples, attr):
	gba = samples.groupby(attr).groups.values() #group by attribute
	pa = np.array(list(map(len, gba))) 
	pa = pa / sum(pa) # get the probability of eigenvalue
	hda = map(lambda i:entropy(samples.loc[i]), gba) 
	hda = np.array(list(hda)) #get the eigenvalue sequence of attribution
	return entropy(samples) - sum(hda * pa) # GDA = HD - HDA
def c4_5(samples, attr):
	splitInfo = entropy(samples, attr)
	return id3(samples, attr) / splitInfo if splitInfo != 0 else 0 # gRatioDA = GDA / SplitInfoDA
def gini(samples):
	gb = samples.groupby('y').groups.values()
	p = np.array(list(map(len, gb)))
	p = p / sum(p)
	return 1 - sum(p * p)
def cart(samples, attr):
	gba = samples.groupby(attr).groups.values()
	p = np.array(list(map(len, gba)))
	p = p / sum(p)
	g = map(lambda i:gini(samples.loc[i]), gba)
	g = np.array(list(g))
	return -sum(p * g)
# def pluralityVal(samples):
# 	gb = samples.groupby('y').groups
# 	gbk = list(gb.keys())
# 	gbv = gb.values()
# 	n = np.array(list(map(len, gbv)))
# 	n = global_var['p_y'] * n
# 	return  gbk[0] if n[0] > n[1] else gbk[1]
def pluralityVal(samples):    
	return samples['y'].mode()[0] 
def decisionTreeLearning(samples, attributes, parent_samples=None,importance=c4_5):
	if len(samples)==0: #samples is empty
		return {'leaf':True, 'y':pluralityVal(parent_samples)}
	gb_y = samples.groupby('y')
	if len(gb_y) == 1: #all samples have the same classification
		return {'leaf':True, 'y':list(gb_y.groups.keys())[0]}
	attr = attributes.keys()
	if len(attr) == 0: #attributes is empty
		return {'leaf':True, 'y':pluralityVal(samples)}
	# importance = id3 #choose the determination strategy
	im = list(map(lambda a:importance(samples,a), attr)) #evaluation of each atrribution in strategy
	a = reduce(lambda x,y: x if x[0]>=y[0] else y, zip(im,attr))[1] #determine the next attribution
	ag = range(attributes.pop(a)) #the range of the next attribution
	tree = {'leaf':False, 'attr':a, 'child':[]}
	for v in ag:
		exa = samples[samples[a]==v]
		tree['child'] += [decisionTreeLearning(exa, attributes, samples,importance)]
	return tree
def heuristic_c4_5Tree(samples, attributes, parent_samples=None,importance=None):
	if len(samples)==0: #samples is empty
		return {'leaf':True, 'y':pluralityVal(parent_samples)}
	gb_y = samples.groupby('y')
	if len(gb_y) == 1: #all samples have the same classification
		return {'leaf':True, 'y':list(gb_y.groups.keys())[0]}
	attr = attributes.keys()
	if len(attr) == 0: #attributes is empty
		return {'leaf':True, 'y':pluralityVal(samples)}
	preid3 = list(map(lambda a:id3(samples,a), attributes))
	if len(attributes) > 3:
		li = sorted(zip(preid3, attributes), key = lambda x:x[0], reverse=True)
		res = zip(map(lambda x: x[0]/entropy(samples,x[1]), li[:-3]), map(lambda x: x[1], li[:-3]))
		# a = sorted(res, key=lambda x:x[0], reverse=True)
	else:
		li = list(map(lambda a:c4_5(samples,a), attr)) #evaluation of each atrribution in strategy
		res = zip(li,attr)
	a = reduce(lambda x,y: x if x[0]>=y[0] else y, res)[1] #determine the next attribution
	ag = range(attributes.pop(a)) #the range of the next attribution
	tree = {'leaf':False, 'attr':a, 'child':[]}
	for v in ag:
		exa = samples[samples[a]==v]
		tree['child'] += [heuristic_c4_5Tree(exa, attributes, samples)]
	return tree
def evaluatePrune(tree, valid):
	res = np.array(list(map(lambda x:decide(tree,x[1]), valid.iterrows()))) #take out the valid in line
	std = np.array(valid['y'])
	# print(res, std)
	same = (res == std)
	return same.sum() / len(same) #accuracy
def postPrune(tree, subtree, valid):
	if subtree['leaf']:
		return True
	flag = True
	for node in subtree['child']:
		flag &= postPrune(tree, node, valid) #if prune last time, it can prune this time
	if flag: #prune if raise accuracy else not prune 
		prev_score = evaluatePrune(tree, valid)
		y = map(lambda n: n['y'], subtree['child'])
		subtree['leaf'] = True
		_attr = subtree.pop('attr')
		_next = subtree.pop('child')
		subtree['y'] = Counter(y).most_common()[0][0]
		cur_score = evaluatePrune(tree, valid)
		if cur_score < prev_score: #recover the tree
			subtree['leaf'] = False
			subtree.pop('y')
			subtree['attr'] = _attr
			subtree['child'] = _next
			return False
		else:
			return True
	return False
def decide(tree, example):
	while not tree['leaf']:
		tree = tree['child'][example[tree['attr']]]
	return tree['y']
def evaluate(TP, FN, TN, FP):
	Accuracy = (TP + TN) / np.float64(TP + FP + TN + FN)
	Recall = TP / np.float64(TP + FN)
	Precision = TP/ np.float64(TP + FP)
	F1 = (2 * Precision * Recall) / np.float64(Precision + Recall)
	return (Accuracy, Recall, Precision, F1)
def check(res, std):
	count = [0]*4
	for r, s in zip(res, std):
		wrong = int(r != s)
		neg = int(s != 1)
		count[neg * 2 + wrong] += 1
	return evaluate(*count)
def groupValidation(train, k=10): #get the validation by separating the train
	gb_id = train.groupby('y').groups.values()#divided in two groups (1,-1)
	# p_y = 1/ np.array(list(map(len, gb_id)))
	# p_y /= sum(p_y)
	# global_var['p_y'] = p_y
	L = []
	for g in gb_id:
		n = len(g)
		m = math.ceil(n/k) #divide in k parts
		id_gbid = list(range(n))
		random.shuffle(id_gbid) #scatter in group 1(-1)
	# 	li = []
	# 	for i in range(k):
	# 		li += [list(map(lambda x: g[x], id_gbid[i*m:(i+1)*m]))]
	# 	L += [li]
	# print(L)
		L += [list(map(lambda y:list(map(lambda x:g[x], id_gbid[y:y+m])), map(lambda i:i*m,range(k))))]	
	return list(map(operator.add, *L))
def crossValidation(train, attributes, treefunc=decisionTreeLearning, choose=c4_5):
	gval = groupValidation(train)
	subforest = []
	for i in range(len(gval)): #len(gval)=k
		cp_gval = copy.deepcopy(gval)
		_valid = train.loc[cp_gval.pop(i)] #take out validation i
		_std = list(_valid['y'])
		_valid.drop('y', axis=1)
		_train = train.loc[reduce(operator.add, cp_gval)] # 1 for validation, 9 for train

		########################## ordinary data  ###########################
		Tree = treefunc(_train, copy.deepcopy(attributes))
		_res = list(map(lambda x:decide(Tree, x[1]), _valid.iterrows()))
		cv_res = check(_res, _std)
		if cv_res[0] > 0.4 and cv_res[3] > 0.4: # evaluate standard
			subforest += [Tree]
	return subforest
def pruneTree(train, attributes):
	gval = groupValidation(train)
	_valid = train.loc[reduce(operator.add, gval[:3])] #3 for validation, 7 for train 
	_std = list(_valid['y'])
	# _valid.drop('y', axis=1)
	_train = train.loc[reduce(operator.add, gval[3:])]
	dcTree = decisionTreeLearning(_train, copy.deepcopy(attributes))
	postPrune(dcTree, dcTree, _valid)
	_res = list(map(lambda x:decide(dcTree, x[1]), _valid.iterrows()))
	cv_res = check(_res, _std)
	# print(cv_res)
	return (dcTree,cv_res)
def discretization(data): 
	col = data[0]
	col[col<=30] = 0
	col[(col>30)&(col<=40)] = 1
	col[col>40] = 2
	data[0] = col	

#read the train and test data
file = open('train.csv')
train = pd.read_csv(file, header=None)
file.close()
file = open('test.csv')
test = pd.read_csv(file, header=None)
file.close()

#data discretization
# discretization(train)
# discretization(test)

#change the attributes name in {x1,x2,x3...,x9} and the label in y
attr = list(map(lambda x:'x'+str(x), range(1,10)))
train.columns = test.columns = attr + ['y']
#get the bound of the attributes
train_bound = list(map(lambda x:x+1, train.drop('y',axis=1).max()))
test_bound = list(map(lambda x:x+1, test.drop('y',axis=1).max()))
bound = list(map(lambda x,y:x if x >y else y ,train_bound, test_bound))
#bingding the bound in attributes
attributes = dict(zip(attr, bound))
# print(attributes)

_valid = train.loc[groupValidation(train)[0]]
_std = list(_valid['y'])
_valid.drop('y', axis=1)

# print('decisionTreeLearning:\n')
dcforest = crossValidation(train, attributes, decisionTreeLearning)
# print('heuristic_c4_5Tree:\n')
heurforest = crossValidation(train, attributes, heuristic_c4_5Tree)

# print('pruneTreeLearning:\n')
pruneforest = [pruneTree(train, attributes) for i in range(120)]
pruneforest = list(map(lambda t:t[0] ,filter(lambda x: x[1][0] > 0.4 and x[1][3] > 0.4, pruneforest)))
forest = dcforest + heurforest + pruneforest
print('-------------'+str(len(forest))+'-------------')

_res = list(map(lambda f: list(map(lambda x:decide(f,x[1]), _valid.iterrows())), forest))
va_res =np.array(_res).sum(axis=0)
va_res[va_res >= 0] = 1
va_res[va_res < 0] = -1
print(check(va_res, _std))

forest_test = list(map(lambda f: list(map(lambda x:decide(f,x[1]), test.iterrows())), forest))
res_test =np.array(forest_test).sum(axis=0)
res_test[res_test >= 0] = 1
res_test[res_test < 0] = -1
print(len(res_test))


# file = open('15352116_hongziqi.txt','w')
# for i in res_test:
# 	file.write(str(i)+'\n')
# file.close()

##############################       draw           ####################################
# dcforest = crossValidation(train, attributes, decisionTreeLearning)
# pl.figure()
# id3pl = pl.plot(list(map(lambda x:x[0], global_var[id3])),'r', label='id3')
# # c4_5pl = pl.plot(list(map(lambda x:x[0],global_var[c4_5])),'b',label='c4.5')
# # cartpl = pl.plot(list(map(lambda x:x[0],global_var[cart])),'g', label='cart')
# pl.title('compare decision Tree in discretization')
# pl.xlabel('times')
# pl.ylabel('Accuracy')
# pl.xlim(0.0, 90.0)
# pl.ylim(0.0, 1.0)
# pl.legend()
# pl.show()

############################### tese little train   ####################################
# file = open('littletrain.csv')
# Ltrain = pd.read_csv(file, header=None)
# file.close()
# file = open('littletest.csv')
# Ltest = pd.read_csv(file, header=None)
# file.close()
# Lattr = list(map(lambda x:'x'+str(x), range(1,3)))
# Ltrain.columns = Ltest.columns = Lattr + ['y']
# Ltrain_bound = list(map(lambda x:x+1, Ltrain.drop('y',axis=1).max()))
# Ltest_bound = list(map(lambda x:x+1, Ltest.drop('y',axis=1).max()))
# Lbound = list(map(lambda x,y:x if x >y else y ,Ltrain_bound, Ltest_bound))
# Lattributes = dict(zip(Lattr, Lbound))

# choose = id3
# print('id3 : ')
# LTree = decisionTreeLearning(Ltrain, copy.deepcopy(Lattributes),importance=choose)
# print(LTree)
# choose = c4_5
# print('c4.5 : ')
# LTree = decisionTreeLearning(Ltrain, copy.deepcopy(Lattributes),importance=choose)
# print(LTree)
# choose = cart
# print('cart : ')
# LTree = decisionTreeLearning(Ltrain, copy.deepcopy(Lattributes),importance=choose)
# print(LTree)

# res = list(map(lambda x:decide(LTree, x[1]), Ltest.iterrows()))
# print(res)


