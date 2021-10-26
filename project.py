import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import model_selection

data = pd.read_csv('mammographic_masses.data.txt', na_values = ["?"], header = None)
df = data.rename(columns = {0: "Bi-Rads", 1: 'Age', 2: 'Shape', 3: "Margin", 4: "Density", 5: "Severity"})
cleaned_df = df.fillna(df.mean())

scaler = preprocessing.MinMaxScaler()

#Decision Tree
from sklearn import tree

features = list(cleaned_df.columns[1:5])
y = cleaned_df["Severity"]
X = cleaned_df[features]
X = scaler.fit_transform(X)
clf = tree.DecisionTreeClassifier()
clf=clf.fit(X,y)

#Tree visualization
from six import StringIO
import pydot
dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data, feature_names = features)
graph, = pydot.graph_from_dot_data(dot_data.getvalue())
graph.write_png('doctree.png')

#Validation
scores = model_selection.cross_val_score(clf, X, y, cv=8)
print("Decision tree:")
print(scores.mean())

#Random Forest
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=11)
clf = clf.fit(X, y)
#Validation
scores = model_selection.cross_val_score(clf, X, y, cv=8)
print("Random forest:")
print(scores.mean())

#KNN
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors = 10)
neigh.fit(X, y)
#KNN Validation
scores = model_selection.cross_val_score(neigh, X, y, cv=8)
print("KNN")
print(scores.mean())

#SVM
from sklearn import svm

C = 0.5
svc = svm.SVC(kernel = 'poly', C=C).fit(X,y)

#SVM validaiton
scores = model_selection.cross_val_score(svc, X, y, cv=8)
print("SVM")
print(scores.mean())

 #naive_bayes
from sklearn.naive_bayes import MultinomialNB

classifier = MultinomialNB()
classifier.fit(X, y)
#Bayes Validation
scores = model_selection.cross_val_score(classifier, X, y, cv=8)
print("Naive_Bayes:")
print(scores.mean())
