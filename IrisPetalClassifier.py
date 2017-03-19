
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn import graph
from sklearn.externals.six import StringIO
from IPython.display import Image

import pydot
import numpy as np

iris = load_iris()
print iris.feature_names
print iris.target_names
print iris.data[0]
print iris.target[0]

for i in range(len(iris.target)):
    print "Example %d: label %d features %s"%(i,iris.target[i],iris.data[i])

#get the test set from the train data and choose 5 for test set
test_idx= [0,50,100]
train_target = np.delete(iris.target,test_idx)
train_data = np.delete(iris.data,test_idx,axis = 0)

test_target = iris.target[test_idx]
test_data = iris.data[test_idx]
#initialise a classifier
clf = tree.DecisionTreeClassifier()

#fit parameters for train data
clf.fit(train_data,train_target)
print test_target
#predict using classifier
predictions =clf.predict(train_data)
a = accuracy_score(train_target,predictions)
print a*100

#get the decision tree in a pdf using graphViz

dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data,
                         feature_names=iris.feature_names,
                         class_names=iris.target_names,
                         filled=True, rounded=True,
                         impurity=False)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("iris.pdf")

print train_target
print predictions
