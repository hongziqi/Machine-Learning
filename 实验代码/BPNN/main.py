import numpy as np
import pandas as pd
import BPNN
import random
import matplotlib.pyplot as plt
# from BPNN import BilayerCModel as bpm

def bitwise(x, n):
	bit = []
	while n > 0:
		bit += [x & 1]
		x >>= 1
		n -= 1
	return bit
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

train_df['date'] = train_df['dteday'].apply(lambda x:int(x.split('/')[-1]))
test_df['date'] = test_df['dteday'].apply(lambda x:int(x.split('/')[-1]))

###########
plt.figure(1)
plt.subplot(211)
plt.plot(train_df['temp'], label='temperature', color='b')
plt.legend()
plt.subplot(212)
plt.plot(train_df['atemp'], label='feeling temperature', color='g')
plt.legend()

plt.figure(2)
plt.subplot(211)
plt.plot(train_df['hum'], label='humidity', color='b')
plt.legend()
plt.subplot(212)
plt.plot(train_df['windspeed'], label='wind speed', color='g')
plt.legend()


for col in ['temp', 'atemp', 'hum', 'windspeed']:
	train_df[col] = BPNN.scale(train_df[col])
	test_df[col] = BPNN.scale(test_df[col])

plt.figure(3)
plt.subplot(211)

plt.plot(train_df['temp'], label='temperature', color='b')
plt.legend()
plt.subplot(212)
plt.plot(train_df['atemp'], label='feeling temperature', color='g')
plt.legend()

plt.figure(4)
plt.subplot(211)
plt.plot(train_df['hum'], label='humidity', color='b')
plt.legend()
plt.subplot(212)
plt.plot(train_df['windspeed'], label='wind speed', color='g')
plt.legend()
plt.show()

########################
season = pd.get_dummies(train_df['season'],prefix='season')
weathersit = pd.get_dummies(train_df['weathersit'],prefix='weathersit')
weekday = pd.get_dummies(train_df['weekday'],prefix='weekday')
mnth = pd.get_dummies(train_df['mnth'],prefix='mnth')

train_df = pd.concat([train_df,season],axis=1)
train_df = pd.concat([train_df,weathersit],axis=1)
train_df = pd.concat([train_df,weekday],axis=1)
train_df = pd.concat([train_df,mnth],axis=1)

season = pd.get_dummies(test_df['season'],prefix='season')
weathersit = pd.get_dummies(test_df['weathersit'],prefix='weathersit')
weekday = pd.get_dummies(test_df['weekday'],prefix='weekday')
mnth = pd.get_dummies(test_df['mnth'],prefix='mnth')

test_df = pd.concat([test_df,season],axis=1)
test_df = pd.concat([test_df,weathersit],axis=1)
test_df = pd.concat([test_df,weekday],axis=1)
test_df = pd.concat([test_df,mnth],axis=1)

# train_df['season0'], train_df['season1'] = map(list, zip(*(train_df['season']-1).apply(lambda x:bitwise(x, 2))))
# train_df['weathersit0'], train_df['weathersit1'] = map(list, zip(*(train_df['weathersit']-1).apply(lambda x:bitwise(x, 2))))
# train_df['weekday0'], train_df['weekday1'], train_df['weekday2'] = map(list, zip(*train_df['weekday'].apply(lambda x:bitwise(x, 3))))
# train_df['mnth0'], train_df['mnth1'], train_df['mnth2'], train_df['mnth3'] =  map(list, zip(*train_df['mnth'].apply(lambda x:bitwise(x, 4))))
train_df['hr0'], train_df['hr1'], train_df['hr2'], train_df['hr3'], train_df['hr4'] =  map(list, zip(*train_df['hr'].apply(lambda x:bitwise(x, 5))))

# test_df['season0'], test_df['season1'] =  map(list, zip(*(test_df['season']-1).apply(lambda x:bitwise(x, 2))))
# test_df['weathersit0'], test_df['weathersit1'] =  map(list, zip(*(test_df['weathersit']-1).apply(lambda x:bitwise(x, 2))))
# test_df['weekday0'], test_df['weekday1'], test_df['weekday2'] =  map(list, zip(*test_df['weekday'].apply(lambda x:bitwise(x, 3))))
# test_df['mnth0'], test_df['mnth1'], test_df['mnth2'], test_df['mnth3'] =  map(list, zip(*test_df['mnth'].apply(lambda x:bitwise(x, 4))))
test_df['hr0'], test_df['hr1'], test_df['hr2'], test_df['hr3'], test_df['hr4'] =  map(list, zip(*test_df['hr'].apply(lambda x:bitwise(x, 5))))


useless = ['instant', 'date','dteday', 'season', 'mnth', 'hr', 'weekday', 'weathersit', 'cnt']
# useless = ['instant', 'dteday', 'cnt']
train_X = train_df.drop(useless, axis=1).as_matrix()
train_y = train_df[['cnt']].as_matrix()
test_X = test_df.drop(useless, axis=1).as_matrix()
vali_X = train_X[-475:]
vali_y = train_y[-475:]

train_X = train_X[:-475]
train_y = train_y[:-475]
# print(train_y)
model = BPNN.Batch_BilayerRModel(train_X, train_y, vali_X, vali_y, hwidth=10 ,times=500)
test_y = model.predict(test_X)
print(test_y)
test_df['cnt'] = np.round(test_y)
print(test_df['cnt'])



# ###########validation########
num = random.sample(range(len(train_X)), 700)
index = np.array([True] * len(train_X))
index[num] = False
vali_X, vali_y = train_X[num], train_y[num]
train_X, train_y = train_X[index], train_y[index]

vali_X = train_X[-475:]
vali_y = train_y[-475:]

train_X = train_X[:-475]
train_y = train_y[:-475]
model = BPNN.Batch_BilayerRModel(train_X, train_y, vali_X, vali_y, hwidth=10 ,times=500)

# ############plot#############
loss = model.loss()
# print(loss)
plt.subplot(111)
plt.plot(range(len(loss['train'])), loss['train'], label='Training loss')
plt.plot(range(len(loss['train'])), loss['validation'], label='validation loss')
plt.legend()

######little validation######
pred_vali_y = model.predict(vali_X)

plt.subplot(212)
plt.plot(pred_vali_y, label='Prediction')
plt.plot(vali_y, label='Data')
plt.xlim(xmax=len(vali_y))
plt.legend()

plt.show()


##############################Little ################
little_df = pd.read_csv('litrain.csv')
little_X = little_df.drop(['cnt'], axis=1).as_matrix()
little_y = little_df[['cnt']].as_matrix()
print(little_y)
model = BPNN.Little(little_X, little_y)



# train_df = pd.concat([train_df,target],axis=1)
