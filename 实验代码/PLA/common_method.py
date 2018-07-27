import numpy as np
import csv

def getData(filestring, target=False, predict=False):
	dataset = []
	file = open(filestring)
	for line in file:
		item = {}
		col = line.strip('\n').split(',')
		# item['x'] = [1] + list(map(float,col[:-1]))
		if target:
			item['x'] = [1] + list(map(float,col[:-1]))
			item['y'] = int(col[-1])
		if predict:
			item['x'] = [1] + list(map(float,col[:]))
			item['y_hat'] = 1
		dataset += [item]
	file.close()
	return dataset

def sign(a):
	return 1 if a > 0 else -1

def check(test, weights):
	count = [0] * 4
	for item in test:
		wTx = sum(list(map(lambda x: x[0]*x[1], zip(item['x'], weights))))
		wrong = int(sign(wTx) != item['y'])
		neg = int(item['y'] != 1)
		count[neg * 2 + wrong] += 1
	TP, FN, TN, FP = count
	print("[TP, FN, TN, FP] :")
	print(count)
	Ac, Re, Pre, F1 = evaluate(TP, FN, TN, FP)
	print( 'Ac: %s\t Re: %s\t Pre: %s\t F1: %s' % (Ac, Re, Pre, F1) )
	return (Ac, Re, Pre, F1)

def evaluate(TP, FN, TN, FP):
	Accuracy = (TP + TN) / np.float64(TP + FP + TN + FN)
	Recall = TP / np.float64(TP + FN)
	Precision = TP/ np.float64(TP + FP)
	F1 = (2 * Precision * Recall) / np.float64(Precision + Recall)
	return (Accuracy, Recall, Precision, F1)


def getResult(test,w,filestring):
	for item in test:
		item['y_hat'] = sign(sum(item['x']*w))
	fileO = open(filestring, 'w', newline='')
	writer = csv.writer(fileO)
	for i in range(len(test)):
		itemres = test[i]['y_hat']
		print(itemres)
		writer.writerow([str(itemres)])
	fileO.close()
	return 

def save_weights(fname, weights):
	file = open(fname, 'w')
	data_str = reduce(lambda x,y:str(x)+','+str(y), weights)
	file.write(data_str + '\n')
	file.close()

def load_weights(fname):
	file = open(fname)
	col = file.readline().strip('\n').split(',')
	file.close()
	return list(map(float, col))
