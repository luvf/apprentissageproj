import numpy as np

from  read_data import *

from utils import to_class, class_to_number, multiclass_log_loss

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier

<<<<<<< HEAD
=======
file = "../../train.csv"
#data = 	prepare(file)

data = full_train(file)
pr= get_predict("../../test.csv")

###Principal composent analysis
from sklearn.decomposition import PCA
pca = PCA(n_components=50)
def reducted_train(file):
    d= get_datas(file)
    features = get_features(d)
    new_features = pca.fit_transform(features)
    return new_features
rdata = reducted_train(file)

#### Naive Bayes
'''
from sklearn.naive_bayes import GaussianNB
>>>>>>> b9b9474c298f6600872bd9e073e0ae10c755214b

from sklearn.datasets import make_classification
from sklearn.naive_bayes import MultinomialNB

from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import tree
from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler
from mlxtend.classifier import StackingClassifier,StackingCVClassifier

import xgboost as xgb
file = "../train.csv"




svc = (lambda: SVC(gamma='auto'))
# relativement lent, score 0.571

MLP = (lambda : MLPClassifier(solver='adam', activation ='relu', max_iter = 300, hidden_layer_sizes=(70)))
MLP2 = (lambda : MLPClassifier(solver='adam', activation ='logistic', max_iter = 300, hidden_layer_sizes=(65,40)))

RandomForest = (lambda : RandomForestClassifier(n_estimators=300, max_depth=150,random_state=0))
GradientBoost = (lambda : GradientBoostingClassifier(loss = 'deviance', n_estimators=800,  subsample= 0.9, criterion = "friedman_mse"))
#Knlambda = (lambda : KNeighborsClassifier(n_neighbors=100, weights='distance',algorithm = 'auto', p=2))
logistic = (lambda : LogisticRegression(solver='newton-cg', multi_class='multinomial'))

def create(x, l):
	"""not used with bagging  classifier"""

	out=[]
	for i in range(x):
		out.append(l())
	return out

def Stacking():
	estimators= list()
	estimators+= [BaggingClassifier(RandomForest(),n_estimators= 5)]
	estimators+= [BaggingClassifier(GradientBoost(),n_estimators= 2)]
	#ll++ [xgb.XGBClassifier(n_estimators=300,max_depth=150)]
	estimators+= [BaggingClassifier(MLP(),n_estimators= 10, bootstrap_features=True)]
	estimators+= [BaggingClassifier(MLP2(),n_estimators= 10)]

	return StackingClassifier(estimators, use_probas=True,average_probas=True, verbose =True,use_features_in_secondary= True, meta_classifier=logistic())

def simple_stacing():
	estimators= list()
	estimators.append(RandomForest())
	estimators.append(GradientBoost())
	estimators.append(MLP())
	estimators.append(MLP2())

	return StackingClassifier(estimators, use_probas=True, verbose =True, meta_classifier=logistic())




def Voting():
	estimators=[]
	estimators.append(("e1", GradientBoost()))
	estimators.append(("e2", RandomForest()))
	estimators.append(("e3", MLP()))
	estimators.append(("e4", MLP2()))
	return VotingClassifier(estimators , voting='soft')



def train_fullset(model,outfile):
	data = full_train(file)
	pr = get_predict("../test.csv")
	print("donnés chargés")
	model.fit(data[0], list(data[1]))
	print("modele apris")
	y_pred_proba = model.predict_proba(pr)
	write_output_proba(y_pred_proba, outfile)
	return model, y_pred_proba

def train_test(model):
	data = prepare(file)
	print("donnés chargés")
	model.fit(data[0][0], list(data[0][1]))
	print("modèle appris")

	y_pred_proba = model.predict_proba(data[1][0])
	y_pred = model.predict(data[1][0])
	
	y_true = class_to_number(data[1][1])
	print("Number of mislabeled points out of a total %d points : %d" % ( len(data[1][1]),(data[1][1] != y_pred).sum()))
	score = (data[1][1] != y_pred).sum()

	print("erreur : ", score/len(data[1][1]))
	print("log-loss : ", multiclass_log_loss(y_true,y_pred_proba))	
	return model, y_pred, y_pred_proba, y_true


model = Voting()
model, pred_prob = train_fullset(gnb, "dontstopmenow.csv")
model, pred, probs, t =train_test(gnb)

