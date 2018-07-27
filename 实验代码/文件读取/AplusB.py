def AplusB(strtriadA,strtriadB):
	triadA = []
	triadB = []
	fileI = open(strtriadA)
	for line in fileI:
		str0 = line[1:line.rfind(']')]
		triadA.append(str0.split(','))
	fileI.close()
	# print(triadA)

	fileI = open(strtriadB)
	for line in fileI:
		str0 = line[1:line.rfind(']')]
		triadB.append(str0.split(','))	
	fileI.close()
	triadC = []
	
	if triadA[0][0] != triadB[0][0] or triadA[1][0] != triadB[1][0]:
		fileO = open('triadC.txt','w')
		fileO.write('The dimension is different!')
		fileO.close()
		return
	else:	
		for i in range(3,len(triadA)):
			for j in range(3,len(triadB)):
				if triadA[i][0] == triadB[j][0] and triadA[i][1] == triadB[j][1]:
					triadA[i][2] = str(float(triadA[i][2]) + float(triadB[j][2])) 
					triadB.pop(j)
					break
		
		for i in range(3,len(triadA)):
			triadA[i][2] = round(float(triadA[i][2]),4)
		for i in range(3,len(triadB)):
			triadB[i][2] = round(float(triadB[i][2]),4) 

		for i in range(3,len(triadA)):
			if triadA[i][2] != 0:
				triadC.append(triadA[i])
		for i in range(3,len(triadB)):
			triadC.append(triadB[i])
		triadC.sort()
		fileO = open('triadC.txt','w')
		fileO.write('['+str(triadA[0][0]) + ']\n')
		fileO.write('['+str(triadA[1][0]) + ']\n')
		fileO.write('[' + str(len(triadC)) +']\n')
		for i in range(0,len(triadC)):
			fileO.write('['+str(triadC[i][0])+','+str(triadC[i][1])+','+str(triadC[i][2]) + ']\n')
		fileO.close()
		
		return



	