fileI = open('onehot.txt')

data = []
row = 0
count = 0
for line in fileI:
	str0 = line[1:line.rfind(']')]
	item = str0.split(', ')
	data.append(item)
	row += 1
	for i in range(0,len(item)):
		if int(item[i]) != 0:
			count += 1
col = len(data[0])
fileI.close()

fileO = open('smatrix.txt','w')
fileO.write('[' + str(row) + ']\n')
fileO.write('[' + str(col) + ']\n')
fileO.write('[' + str(count) + ']\n')

for i in range(0,row):
	for j in range(0,col):
		if int(data[i][j]) != 0:
			fileO.write('[' + str(i) + ', ' + str(j) + ', ' + str(data[i][j]) + ']\n') 

fileO.close()