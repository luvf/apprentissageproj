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
	for i in range(len(labels)):
		labels[i] = int(labels[i][-1])
	return labels

def write_output(data):
	labesl =["Class_1","Class_2","Class_3","Class_4","Class_5","Class_6","Class_7","Class_8","Class_9"]



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

def split_non_det(data):
	"""TODO"""
	return 
