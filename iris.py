from sklearn.datasets import load_iris
import numpy as np
from sklearn import tree

iris = load_iris()
testing_idx = [0,50,100]

'''
print(iris.feature_names)
print(iris.target_names)
print(iris.data[0])
print(iris.target[0])

for i in range(len(iris.data))
	print("example %d: features: %s label: %s" %(i,iris.data[i],iris.target[i]))

'''

# training data
training_target = np.delete(iris.target,testing_idx)
training_data = np.delete(iris.data,testing_idx, axis=0)

# testing data
test_data = iris.data[testing_idx]
test_target = iris.target[testing_idx]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(training_data,training_target)

print(test_target)
print(clf.predict(test_data))

# viz
from sklearn.externals.six import StringIO 
import pydotplus

dot_data = StringIO()
tree.export_graphviz(
	clf, out_file=dot_data,
	feature_names=iris.feature_names,
	class_names=iris.target_names,
	filled=True, rounded = True,
	impurity=False)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("iris.pdf")

print(test_data[2], test_target[2])

print(iris.feature_names,iris.target_names)