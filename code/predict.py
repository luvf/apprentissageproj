import numpy as np

from  read_data import *

from sklearn.metrics import log_loss




file = "../train.csv"
data = 	prepare(file)

#data = full_train(file)
#pr= get_predict("../test.csv")

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
from sklearn.svm import SVC
gnb = SVC(gamma='auto')

from sklearn import tree
gnb = tree.DecisionTreeClassifier()


from sklearn.neural_network import MLPClassifier
gnb = MLPClassifier(solver='lbfgs', alpha=1e-5,  hidden_layer_sizes=(70, 50, 30), random_state=1)


#gnb = nonML()
#gnb = GaussianNB()
gnb.fit(data[0][0], list(data[0][1]))

y_pred = gnb.predict(data[1][0])
y_pred_proba = gnb.predict_proba(data[1][0])


#### Random Forest
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.datasets import make_classification

#rndf = RandomForestClassifier(n_estimators=100, max_depth=20,random_state=0)
#y_pred = rndf.fit(data[0][0], list(data[0][1])).predict(data[1][0])


#y_pred = gnb.fit(data[0], list(data[1])).predict(pr)
####




#write_output(y_pred)


score = (data[1][1] != y_pred).sum()
print("erreur : ", score/len(data[1][1]))


def to_class(l):
	out = np.zeros(l.shape)
	for i in range(out.shape[0]):
		out[i] =  int(l[i][-1])-1
	return out

def class_to_number(class_list):
    number_list = np.zeros([len(class_list)])
    for i in range(len(class_list)):
        number_list[i] = int(class_list[i][-1:])-1
    return number_list


def multiclass_log_loss(y_true, y_predict, eps=1e-15):
    predictions = np.clip(y_predict, eps, 1 - eps)
    # normalize row sums to 1
    predictions /= predictions.sum(axis=1)[:, np.newaxis]

    actual = np.zeros(y_predict.shape)
    rows = actual.shape[0]
    actual[np.arange(rows), y_true.astype(int)] = 1
    vsota = np.sum(actual * np.log(predictions))
    return -1.0 / rows * vsota



y_true_temp = data[1][1]
y_true = class_to_number(y_true_temp)
print("Number of mislabeled points out of a total %d points : %d" % ( len(data[1][1]),(data[1][1] != y_pred).sum()))
print(multiclass_log_loss(y_true,y_pred_proba))


class nonML:
	def __init__(self):
		self.aa =0

	def fit(self, data, labels):
		self.data = data
		self.labels = labels

	def predict_el(self,el):#on peut faire plus pythonique
		minn = 1000
		for i in range(self.data.shape[0]):
			tmp= np.abs(el-data).sum()
			if tmp < minn:
				ret = i
				minn = tmp
		return self.labels[ret]
	

	def predict(self, els):
		ret= list()
		for el in els:
			ret.append(self.predict_el(el))
		return ret




