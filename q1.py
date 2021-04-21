
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from logisticRegression.logisticRegression import LogisticRegression
from metrics import *
from sklearn.datasets import load_breast_cancer
from sklearn import preprocessing
from sklearn.model_selection import KFold

np.random.seed(42)

def a():

    N = 30
    M = 5 
    X = pd.DataFrame(np.random.randn(N, M))
    y = pd.Series(np.random.randint(0,2,N))
    
    X = preprocessing.StandardScaler().fit_transform(X)


    print(X)
    print(y)

    for fit_intercept in [True,False]:
        LR = LogisticRegression(fit_intercept=fit_intercept)
        LR.fit_non_vectorised(X, y) # here you can use fit_non_vectorised / fit_autograd methods
        y_hat = LR.predict(X)

        # print(np.array(y))
        # print(np.array(y_hat))

        print('Accuracy: ', accuracy(y_hat, y))


def b():


    N = 30
    M = 5 
    X = pd.DataFrame(np.random.randn(N, M))
    y = pd.Series(np.random.randint(0,2,N))

    X = preprocessing.StandardScaler().fit_transform(X)

    print(X)
    print(y)

    for fit_intercept in [True,False]:
        LR = LogisticRegression(fit_intercept=fit_intercept)
        LR.fit_autograd(X, y,n_iter=1000) # here you can use fit_non_vectorised / fit_autograd methods
        y_hat = LR.predict(X)

        # print(np.array(y))
        # print(np.array(y_hat))

        print('Accuracy: ', accuracy(y_hat, y))


def c():

    X,y=load_breast_cancer(return_X_y=True,as_frame=True)
    cols=X.columns
    X = preprocessing.StandardScaler().fit_transform(X)
    Xdf = pd.DataFrame(X,columns=cols)
    k=3

    kf = KFold(n_splits=k)
    LR = LogisticRegression(fit_intercept=True)
    
    acc=[]

    ind = 1



    for train_index, test_index in kf.split(X):
        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        LR.fit_non_vectorised(X_train, y_train, n_iter=1000) # here you can use fit_non_vectorised / fit_autograd methods
        y_hat = LR.predict(X_test)

        acc.append(accuracy(y_hat, y_test))

        print("Fold {}, Accuracy: {}".format(ind,acc[-1]))

        ind+=1


    print("Overall Accuracy: ",np.mean(acc))

    LR.plot_surface(Xdf,y,0,1)


# a()
# b()
c()