'''
Hello World for Machine Learning
'''

from sklearn import tree;

'''
Training Data
'''
'''
Features
'''
smooth = 0;
bumpy = 1;

'''
Labels
'''
Apple = 0;
Oranges = 1;

features = [[140,smooth], [138,smooth], [50, bumpy], [170, bumpy]];
labels = [Oranges, Oranges, Apple, Apple];

clf = tree.DecisionTreeClassifier();

clf = clf.fit(features,labels);

print(clf.predict([[100,bumpy]]));