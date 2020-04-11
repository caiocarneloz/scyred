# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 00:04:11 2020

@author: caiocarneloz
"""
from core import scyred
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

iris = datasets.load_iris()
data = iris.data
labels = iris.target

X_train = data[:15]
y_train = labels[:15]
X_val   = data[15:20]
y_val   = labels[15:20]

input_params = {'max_depth': [3, 10],
                'n_estimators' : [10, 100],
                'ccp_alpha': [0.0, 0.2]}

opt = scyred.optimizer(RandomForestClassifier, input_params, True)
opt.fit(X_train, X_val, y_train, y_val, accuracy_score)
params = opt.run_pso(30, 2, 100, True)

print(params)

model = RandomForestClassifier(**params)
model = model.fit(data, labels)
pred = model.predict(data)

print(accuracy_score(labels, pred))