
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from logisticRegression.kclassLogisticRegression import kclassLogisticRegression
from metrics import *
from sklearn.datasets import load_digits
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
matplotlib_axes_logger.setLevel('ERROR')


np.random.seed(42)

def a():
    X,y=load_digits(return_X_y=True,as_frame=False)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    LR = kclassLogisticRegression(num_classes=10, fit_intercept=True)

    LR.fit_non_vectorised(X_train, y_train, batch_size=50, n_iter=300) # here you can use fit_non_vectorised / fit_autograd methods
    y_hat = LR.predict(X_test)
    
    # print(list(np.array(y_hat)))
    # print(list(np.array(y)))

    print(accuracy(y_hat, y_test))


def c(best_model,X,y):

    X_train, X_test, y_train, y_test = train_test_split(X, np.array(y), test_size=0.33, random_state=42)
    best_model.fit_autograd(X_train, y_train, batch_size=30, n_iter=1000)
    y_pred=np.array(best_model.predict(X_test))

    confusion = confusion_matrix(y_test, y_pred)
    print("\nColumns: Predicted Class, Rows: Actual Class")
    print('Confusion Matrix:')
    print(confusion)


    for matrix in confusion_matrices:
        fig = plt.figure()
        plt.matshow(cm)
        plt.title('Problem 1: Confusion Matrix Digit Recognition')
        plt.colorbar()
        plt.ylabel('True Label')
        plt.xlabel('Predicated Label')
        fig.savefig('confusion_matrix'+str(learning_values.pop())+'.jpg')

    
    print('\nAccuracy: {:.5f}\n'.format(accuracy_score(y_test, y_pred)))
    
    print('\nClassification Report:')
    print(classification_report(y_test, y_pred))
    

    prec,rec,f1,_ = precision_recall_fscore_support(y_test,y_pred)
    d={}
    for i in range(f1.shape[0]):
        d[i] = f1[i]
    l=sorted(d.items(),key=lambda x:x[1],reverse=True)
    print("Most confused digits",end=": ")
    print(l[-1][0],",",l[-2][0])
    print("Easisest digit to predict",end=": ")
    print(l[0][0])
    

def b():
    X,y=load_digits(return_X_y=True,as_frame=True)
    # print(X)
    # print(y)

    cols=X.columns
    X = preprocessing.StandardScaler().fit_transform(X)
    Xdf = pd.DataFrame(X,columns=cols)
    k=4
    # print(X)
    # print(y)

    kf = KFold(n_splits=k)

    acc=[]
    models=[]

    ind = 1


    for train_index, test_index in kf.split(X):
        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # print(X_train.shape)

        LR = kclassLogisticRegression(num_classes=10, fit_intercept=True)

        LR.fit_autograd(X_train, y_train, batch_size=30, n_iter=2000, lr=0.01) # here you can use fit_non_vectorised / fit_autograd methods
        y_hat = LR.predict(X_test)

        acc.append(accuracy(y_hat, y_test))
        models.append([acc[-1],LR])

        print("Fold {}, Accuracy: {}".format(ind,acc[-1]))

        ind+=1


    print("Overall Accuracy: ",np.mean(acc))

    # LR = kclassLogisticRegression(num_classes=10, fit_intercept=True)
    # LR.fit_autograd(X, y, batch_size=50, n_iter=1000) # here you can use fit_non_vectorised / fit_autograd methods
    # y_hat = LR.predict(X)
    # print(accuracy(y_hat, y))

    models=sorted(models,key=lambda x:x[0])
    best_model=models[0][1]

    c(best_model,X,y)

    

def d():

    def get_cmap(n, name='gist_rainbow'):
        '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
        RGB color; the keyword argument name must be a standard mpl colormap name.'''
        return plt.cm.get_cmap(name, n)

    X,y=load_digits(return_X_y=True,as_frame=False)
    
    # print(X.shape)
    # print(y.shape)

    X = preprocessing.StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(X)
    principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
    finalDf = pd.concat([principalDf, pd.DataFrame(y.reshape(-1,1),columns=['target'])], axis = 1)


    fig = plt.figure(figsize = (15,10))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)
    targets = list(range(10))
    # colors = ['r', 'g', 'b']
    cmap = get_cmap(10)
    colors=[cmap(i) for i in range(10)]

    # print(colors)

    for target, color in zip(targets,colors):
        indicesToKeep = finalDf['target'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                , finalDf.loc[indicesToKeep, 'principal component 2']
                , c = color
                , s = 50)
    ax.legend(targets)
    ax.grid()
    plt.savefig('q3_d.png')
    plt.show()

# a()
# b()
d()