import math as np

import copy

fileI = open('semeval')

dictext = {}
ordertext = []
onehot = []
tf = []
D = 0

for line in fileI:
 	l = line.split('\t')
 	words = l[2].split(' ')
 	words[len(words)-1] = (words[len(words)-1].split('\n'))[0]
 	# text = text + words;
 	for oneword in words:
 		if oneword not in dictext:
 			ordertext.append(oneword)
 			dictext[oneword] = 0
 	D += 1
# ordertext = list(set(text))
# ordertext.sort(key = text.index)
# print(ordertext)
# print(D)
fileI.seek(0)

c = 0
count = []
oline = []
tfline = []

for line in fileI:

 	l = line.split('\t')
 	words = l[2].split(' ')
 	words[len(words)-1] = (words[len(words)-1].split('\n'))[0]
 	for oneword in words:
 		dictext[oneword] += 1 
 		c += 1
 	count.append(c)
 	for key in ordertext:
 		oline.append(int(bool(dictext[key])))
 		tfline.append(dictext[key]/c)
 	# print(oline)
 	onehot.append(oline)
 	tf.append(tfline)

 	oline = []
 	tfline = []
 	c = 0
 	for oneword in words:
 		dictext[oneword] = 0

# print(count)
# print(onehot)	
# print(tf)
idf = []
col = len(onehot[0])
for j in range(0,col):
	sigma = 0
	for i in range(0,D):
		if onehot[i][j]:
			sigma += 1
	# print(sigma)
	# base-e logarithmic
	idf.append(np.log2(D/(sigma+1))	)	
# print(idf)
tfidf = []
for i in range(0,D):
	item = copy.copy(tf[i])
	for j in range(0,col):
		item[j] *= idf[j]
	tfidf.append(item) 
# print(tfidf)
fileI.close()

fileO = open('onehot.txt','w')
for i in range(0,D):
	fileO.write(str(onehot[i]))
	# for j in range(0,col):
	# 	fileO.write(str(int(onehot[i][j])) + ' ')
	fileO.write('\n')
fileO.close()

fileO = open('tf.txt','w')
for i in range(0,D):
	fileO.write(str(tf[i]))
	fileO.write('\n')
fileO.close()

fileO = open('tfidf.txt','w')
for i in range(0,D):
	fileO.write(str(tfidf[i]))
	fileO.write('\n')
fileO.close()

