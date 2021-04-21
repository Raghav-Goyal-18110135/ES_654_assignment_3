# logistic regression for feature importance
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from logisticRegression.logisticRegression import LogisticRegression
from metrics import *
from sklearn.datasets import load_breast_cancer
from sklearn import preprocessing
import collections


# define dataset
X,y=load_breast_cancer(return_X_y=True,as_frame=True)
X = preprocessing.StandardScaler().fit_transform(X)
print(X)
print(y)
# define the model
model = LogisticRegression()
# fit the model
model.fit(X, y)
# get importance
importance = model.coef_[0]
# summarize feature importance
d={}
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))
    d[i]=v
d=collections.OrderedDict(sorted(d.items(),key=lambda x:x[1]))
print(d)
# plot feature importance
plt.bar([x for x in range(len(importance))], importance)
plt.show()