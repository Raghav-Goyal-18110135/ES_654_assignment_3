import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from math import sqrt,floor,ceil
# Import Autograd modules here
# import jax.numpy as jnp
# from jax import grad, jit, vmap
from autograd import grad
import autograd.numpy.random as npr
from autograd.misc.optimizers import adam
import autograd.numpy as np
from autograd import elementwise_grad as egrad
from autograd import jacobian

np.seterr(divide='ignore', invalid='ignore')



def forward_pass(W,B, inputs, activations):
    ind=0
    for w, b in zip(W,B):
        outputs = np.dot(inputs, w) + b
        inputs = activations[ind](outputs)
        ind+=1
    # no activation on the last layer
    # W, b = params[-1]
    # return np.dot(inputs, W) + b
    return outputs

def objective(W, B, X, y,activations):

    # print(X.shape)
    # print(y.shape)
    """ Compute the multi-class cross-entropy loss """
    preds = forward_pass(W,B,X,activations)
    preds=np.array(preds).astype(float)
    # print(type(preds))
    # print(preds.shape)
    preds=np.exp(preds)
    predsum=np.sum(preds,axis=1,keepdims=True)
    preds/=predsum
    # print(np.sum(preds,axis=1))
    res = 0
    y=y.squeeze()
    for i in range(preds.shape[0]):
        res -= np.log10(preds[i][int(y[i])])
    res=res/preds.shape[0]
    # print(res)
    return res


class mlp_classifier():

    def __init__(self,sizes,activations):

        self.params=[]
        assert(len(activations)==len(sizes)-1)
        self.activations=[self.identity] 
        for z in activations:
            if z=='relu':
                self.activations.append(self.relu)
            elif z=='sigmoid':
                self.activations.append(self.sigmoid)
            if z=='identity':
                self.activations.append(self.identity)
        scale=0.01
        rs=npr.RandomState(0)
        self.W=[rs.randn(insize, outsize)*scale for insize, outsize in zip(sizes[:-1], sizes[1:])]
        self.B=[rs.randn(outsize)*scale for insize, outsize in zip(sizes[:-1], sizes[1:])]

    def relu(self,X):
        return np.maximum(0,X)

    def sigmoid(self,X):
        return 1/(1+np.exp(-X))

    def identity(self,X):
        return X

    

    # def objective(params, _):
    #     pred = nn_predict(params, X.reshape([-1, 1]))
    #     err = Y.reshape([-1, 1]) - pred
    #     return np.mean(err**2)

    def fit(self, X, y, batch_size=5, n_iter=10000, lr=0.001, lr_type='constant'):
        X=np.array(X).astype(np.float32)
        y=np.array(y).reshape(-1,1).astype(np.float32)

        m,n=X.shape

        epochs=ceil(n_iter/floor(m/batch_size))
        
        # print(epochs)

        # objective_grad = grad(self.objective)

        for i in range(epochs):
            for j in range(0,m,batch_size):

                X_batch=X[j:j+batch_size]
                y_batch=y[j:j+batch_size]
                
                grad_W=egrad(objective,0)
                grad_B=egrad(objective,1)

                grad_W_list=grad_W(self.W,self.B,X_batch,y_batch,self.activations)
                grad_B_list=grad_B(self.W,self.B,X_batch,y_batch,self.activations)

                for i in range(len(grad_W_list)):
                    self.W[i]-=lr*grad_W_list[i]
                    self.B[i]-=lr*grad_NB_list[i]


            print("Epoch: ",i,end=" ")
            print("cost: ",objective(self.W,self.B,X,y,self.activations))

        # print('Training loss %.6f' % self.loss(theta,X,y))

    def predict(self, X):
        m=X.shape[0]
        preds = self.forward_pass(self.params, X)
        # print(preds.shape)
        preds=np.exp(preds)
        predsum=np.sum(preds,axis=1,keepdims=True)
        preds/=predsum

        res=np.argmax(preds,axis=1)
        return pd.Series(res)




def accuracy(y_hat, y):
    """
    Function to calculate the accuracy
    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the accuracy as float
    """
    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    assert(y_hat.size == y.size)
    # TODO: Write here

    y=pd.Series(y)
    y_hat=pd.Series(y_hat)

    ground=y.to_numpy(copy=True)
    pred=y_hat.to_numpy(copy=True)
    n=ground.size
    cor=0
    for i in range(n):
        if ground[i]==pred[i]:
            cor+=1
    return cor/n





import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from MLP.mlp import mlp_classifier, mlp_regressor
# from metrics import *
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



digits()