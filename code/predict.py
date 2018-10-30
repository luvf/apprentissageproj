
from read_data import *




file = "../train.csv"
data = 	prepare(file)


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
from sklearn.svm import SVC
gnb = SVC(gamma='auto')

from sklearn import tree
gnb = tree.DecisionTreeClassifier()


from sklearn.neural_network import MLPClassifier
gnb = MLPClassifier(solver='lbfgs', alpha=1e-5,  hidden_layer_sizes=(100,50), random_state=1)


y_pred = gnb.fit(data[0][0], list(data[0][1])).predict(data[1][0])

#print("Number of mislabeled points out of a total %d points : %d" % (len(data[1][1],(data[1][1] != y_pred).sum())))
score = (data[1][1] != y_pred).sum()
print("erreur : ", score/len(data[1][1]))