
from read_data import *

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
gnb = MLPClassifier(solver='lbfgs', alpha=1e-5,  hidden_layer_sizes=(70,50, 30), random_state=1)


gnb = GaussianNB()

y_pred = gnb.fit(data[0][0], list(data[0][1])).predict(data[1][0])


#y_pred = gnb.fit(data[0], list(data[1])).predict(pr)

#write_output(y_pred)
#print("Number of mislabeled points out of a total %d points : %d" % (len(data[1][1],(data[1][1] != y_pred).sum())))
#print(log_loss(data[1][1],y_pred))

score = (data[1][1] != y_pred).sum()
print("erreur : ", score/len(data[1][1]))


def to_class(l):
	out = np.zeros(l.shape)
	for i in range(out.shape[0]):
		out[i] =  int(l[i][-1])-1
	return out


import numpy as np

def multiclass_log_loss(y_true, y_pred, eps=1e-15):
	predictions = np.clip(y_pred, eps, 1 - eps)

	# normalize row sums to 1
	predictions /= predictions.sum(axis=1)[:, np.newaxis]

	actual = np.zeros(y_pred.shape)
	rows = actual.shape[0]
	actual[np.arange(rows), y_true.astype(int)] = 1
	vsota = np.sum(actual * np.log(predictions))
	return -1.0 / rows * vsota