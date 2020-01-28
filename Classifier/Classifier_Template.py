import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import file
df = pd.read_csv("iris.csv")


# features && label
features = iris.iloc[:, 0:4]
label = iris['species']


#preprocessing on strings
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(label)
label = le.transform(label)


#Data division
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.3, random_state=0)


#Import Algorithm 
from sklearn import tree
classifier = tree.DecisionTreeClassifier()
classifier.fit(X_train, y_train)


#prediction
pred = classifier.predict(X_test)
pred


#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, pred)
cm


#Accuracy
from sklearn.metrics import accuracy_score
accuracy_score(y_test, pred)


# save the model
from joblib import dump,load 
dump(classifier, "iris.joblib")


#Again loading the model 
##classifier = load("iris.joblib")


