import numpy as np
import copy
import random
import operator
import re
import math
from functools import reduce
import MC_MODEL as mc
import matplotlib.pyplot as plt
from collections import Counter
import xgboost as xgb

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
def clean(string):
	string = re.sub('- *[lr]rb *-', ' ', string)
	string = re.sub('(^| )http:[^ ]+', ' ', string)
	string = re.sub('(^| )[A-Za-z]( |$)', ' ', string)
	string = re.sub('(^| )\'[^ ]+', ' ', string)
	string = re.sub('(^| )n\'t ', ' ', string)
	string = re.sub('[^A-Za-z !?/]+', '', string)
	string = re.sub('(^| )[A-Za-z]( |$)', ' ', string)
	string = re.sub('/', ' ', string)
	return string
def readData(fpath):
    file = open(fpath, encoding='UTF-8')

    label, sentences = [], []
    for line in file:
        l = line.split('\t\t')
        label.append(l[0])
        sen = list(map(clean, l[1].split('<sssss>')))
        sen = list(map(lambda x:
            list(filter(lambda y:y not in stopwords, x.split())), sen))
        sen = list(filter(lambda x:x, sen))
        sentences.append(sen if sen else [['qwerty']])
    file.close()
    return label, sentences
def check(res, std):
	ac = (np.array(res) == np.array(std)).sum() / len(res)
	return ac
file = open('wordvec_6B_50d.txt', encoding='UTF-8')
vec, wdvec_dict, order = [], {}, 0
for line in file:
	item = line.strip().split()
	vec.append(list(map(float,item[1:])))
	wdvec_dict[item[0]] = order
	order += 1
file.close()
def w2v(word):
    if (word in wdvec_dict):
        return np.array(vec[wdvec_dict[word]])
    return np.zeros(50)
# label, train_sen = readData('MulLabelTrain.ss')
# _, test_sen = readData('MulLabelTest.ss')
slabel, strain_sen = readData('BigTrain.ss')
v_label, svalid_sen = readData('SmallValid.ss')

# train_x = []
# for ts in train_sen:
#     v = [np.array(list(map(w2v,s))).mean(axis=0) for s in ts]
#     v = np.array(v).mean(axis=0)
#     train_x.append(v)
# train_x = np.array(train_x)
# # tx0 = np.ones((train_x.shape[0], 1))
# # train_x = np.concatenate((tx0, train_x), axis=1)

# test_x = []
# for ts in test_sen:
#     v = [np.array(list(map(w2v,s))).mean(axis=0) for s in ts]
#     v = np.array(v).mean(axis=0)
#     test_x.append(v)
# test_x = np.array(test_x)
# # tx0 = np.ones((test_x.shape[0], 1))
# # test_x = np.concatenate((tx0, test_x), axis=1)


strain_x = []
for ts in strain_sen:
    v = [np.array(list(map(w2v,s))).mean(axis=0) for s in ts]
    v = np.array(v).mean(axis=0)
    strain_x.append(v)
strain_x = np.array(strain_x)
# x0 = np.ones((strain_x.shape[0], 1))
# strain_x = np.concatenate((x0, strain_x), axis=1)
svalid_x = []
for ts in svalid_sen:
    v = [np.array(list(map(w2v,s))).mean(axis=0) for s in ts]
    v = np.array(v).mean(axis=0)
    svalid_x.append(v)
svalid_x = np.array(svalid_x)
# x0 = np.ones((svalid_x.shape[0], 1))
# svalid_x = np.concatenate((x0, svalid_x), axis=1)

################################################ XGB #####################################
D_name = {'LOW':0, 'MID':1, 'HIG':2}
num_y = np.array([list(map(lambda x: D_name[x], slabel))]).T
xg_train = xgb.DMatrix(strain_x, num_y)
xg_test = xgb.DMatrix(svalid_x)
xgb_params = {
    'eta':0.05,
    'max_depth':6,
    'silent':1,
    'num_class':3,
    'objective':'multi:softprob',
    'subsample':0.7,
    'colsample_bytree':1.0
}
model = xgb.train(xgb_params, xg_train, num_boost_round= 1000)
pred_y = model.predict(xg_test)
pred_y = pred_y.argmax(axis=1)
L_name = ['LOW','MID','HIG']
spred = list(map(lambda x:L_name[x], pred_y))
print(check(spred, v_label))