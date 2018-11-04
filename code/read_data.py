# coding: utf-8
import pandas as pd #for manipulating data 
import collections

import random

import numpy as np

def get_datas(fichier):
	data = pd.read_csv(fichier)
	return data.values

def get_features(data):
	return  data[:,1:94]

def get_labels(data):
	labels =  data[:,-1]
	#for i in range(len(labels)):
	#	labels[i] = int(labels[i][-1])
	return labels

def one_hot(i,lab):
	ret = np.zeros(10, dtype = int)
	ret[0]=i+1
	ret[int(lab[-1])]=1
	return ret


import csv
def write_output(labs):
	with open('out.csv', 'w', newline='') as f:
		writer = csv.writer(f)
		labels =["id","Class_1","Class_2","Class_3","Class_4","Class_5","Class_6","Class_7","Class_8","Class_9"]
		writer.writerow(labels)
		for i in range(labs.shape[0]):
			l = (one_hot(i,labs[i]))
			writer.writerow(l)


def write_output_proba(labs,name):
    with open(name, 'w', newline='') as f:
        writer = csv.writer(f)
        labels =["id","Class_1","Class_2","Class_3","Class_4","Class_5","Class_6","Class_7","Class_8","Class_9"]
        writer.writerow(labels)
        l= []
        for i in range(labs.shape[0]):
            l = [i+1]
            l.extend(list(labs[i]))
            writer.writerow(l)


def split_det(data):
	l = len(data)//10
	train = data[0:l*8]
	test = data[l*8: l*9]
	valid = data[l*9]
	return train, test, valid


def prepare(file):
	d = get_datas(file)
	np.random.shuffle(d)
	features = get_features(d)
	labels= get_labels(d)
	trainf, testf, validf =split_det(features)
	trainl, testl, validl =split_det(labels)
	return ((trainf,trainl), (testf,testl),(validf,validl))

def full_train(file):
	d = get_datas(file)
	features = get_features(d)
	labels= get_labels(d)
	return features, labels

def get_predict(file):
	d = get_datas(file)
	features = get_features(d)
	return features

def split_non_det(data):
	"""TODO"""
	return 


"""
a integrer je conaissais pas dict reader,j'ai peut etre utiliser un marteau piqueur
import csv


# In[25]:


trainfile = open("train.csv", "r")
testfile = open("test.csv", "r")


# In[29]:


train = csv.DictReader(trainfile)
test = csv.DictReader(testfile)


# In[28]:

p
"""