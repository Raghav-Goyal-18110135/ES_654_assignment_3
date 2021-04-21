
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from MLP.mlp import mlp_classifier, mlp_regressor
from metrics import *
from sklearn.datasets import load_digits,load_boston
from sklearn import preprocessing   
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from collections import OrderedDict
from sklearn.decomposition import PCA
from matplotlib.axes._axes import _log as matplotlib_axes_logger
import pickle
matplotlib_axes_logger.setLevel('ERROR')

np.random.seed(40)


def digits():

    X,y=load_digits(return_X_y=True,as_frame=False)
    X = preprocessing.StandardScaler().fit_transform(X)
    
    k=3
    kf = KFold(n_splits=k)

    acc=[]
    models=[]

    ind = 1

    for train_index, test_index in kf.split(X):
        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # print(X_train.shape)

        sizes=[64,32,16,10]
        activations=['relu','relu','identity']

        nn = mlp_classifier(sizes,activations)
        nn.fit(X_train, y_train, batch_size=30, n_iter=8000,lr=0.001) # here you can use fit_non_vectorised / fit_autograd methods
        
        y_hat = nn.predict(X_test)
    
        acc.append(accuracy(y_hat, y_test))
        models.append([acc[-1],nn])

        print("Fold {}, Accuracy: {}".format(ind,acc[-1]))

        ind+=1


    print("Overall Average Accuracy: ",np.mean(acc))


    # sizes=[64,32,20,10]
    # activations=['relu']*3

    # nn = mlp_classifier(sizes,activations)
    # nn.fit(X, y, batch_size=20, n_iter=10000,lr=0.0001) # here you can use fit_non_vectorised / fit_autograd methods

    # print(accuracy(nn.predict(X), y))



def boston_housing():

    X,y=load_boston(return_X_y=True)
    X = preprocessing.StandardScaler().fit_transform(X)
    
    k=3
    kf = KFold(n_splits=k)

    acc=[]
    models=[]

    ind = 1

    for train_index, test_index in kf.split(X):
        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # print(X_train.shape)
        sizes=[13,8,4,1]
        activations=['relu','relu','identity']

        nn = mlp_regressor(sizes,activations)
        nn.fit(X_train, y_train, batch_size=30, n_iter=10000,lr=0.001) # here you can use fit_non_vectorised / fit_autograd methods
        y_hat=nn.predict(X_test)
        err=rmse(y_hat,y_test)
    
        acc.append(err)
        models.append([acc[-1],nn])

        print("Fold {}, RMSE: {}".format(ind,acc[-1]))

        ind+=1


    print("Overall Avergae Error: ",np.mean(acc))


    # sizes=[13,10,8,1]
    # activations=['relu','relu','identity']

    # nn = mlp_regressor(sizes,activations)
    # nn.fit(X, y, batch_size=20, n_iter=10000,lr=0.001) # here you can use fit_non_vectorised / fit_autograd methods
    # y_hat=nn.predict(X)
    # print(rmse((y_hat,y)))

digits()
boston_housing()
