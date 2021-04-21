import numpy as onp
import jax.numpy as np
from jax import grad, jit, vmap, value_and_grad
from jax import random
from math import floor,sqrt,ceil
from jax.scipy.special import logsumexp
from jax.experimental import optimizers
import pandas as pd

class mlp_classifier():

    def __init__(self,sizes,activations):
        self.rnkey = random.PRNGKey(1)
        rnkeys=random.split(self.rnkey,len(sizes))
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
        self.scale=0.01
        for n_prev,n_cur,key in zip(sizes[:-1], sizes[1:], rnkeys):
            x,y=random.split(key)
            self.params.append([self.scale*random.normal(x,(n_cur,n_prev)),self.scale*random.normal(y,(n_cur,))])

    def relu(self,X):
        return np.maximum(0,X)

    def sigmoid(self,X):
        return 1/(1+np.exp(-X))

    def identity(self,X):
        return X

    def forward_pass(self,params,X):
        ind=0
        activations = X
        for W,b in params:
            activations=self.activations[ind](np.matmul(activations,W.T)+b)
            ind+=1
        return activations

    def cross_entropy_loss(self, params, X, y):

        # print(X.shape)
        # print(y.shape)
        """ Compute the multi-class cross-entropy loss """
        preds = self.forward_pass(params, X)
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

    def fit(self, X, y, batch_size=5, n_iter=10000, lr=0.001, lr_type='constant'):
        X=np.array(X).astype(np.float32)
        y=np.array(y).reshape(-1,1).astype(np.float32)

        m,n=X.shape

        opt_init, opt_update, get_params = optimizers.adam(lr)
        opt_state = opt_init(self.params)

        epochs=ceil(n_iter/floor(m/batch_size))

        for i in range(epochs):
            for j in range(0,m,batch_size):

                X_batch=X[j:j+batch_size]
                y_batch=y[j:j+batch_size]

                cur_loss, grads = value_and_grad(self.cross_entropy_loss)(self.params,X_batch,y_batch)

                # for z in range(len(self.params)):
                #     self.params[z][0]-=lr*grads[z][0]
                #     self.params[z][1]-=lr*grads[z][1]
                # print(grads[0])
                opt_state = opt_update(0, grads, opt_state)
                self.params=get_params(opt_state)

                # print(cur_loss)
                
            
            print("Epoch: ",i,end=" ")
            print("cost: ",self.cross_entropy_loss(self.params,X,y))

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


class mlp_regressor():

    def __init__(self,sizes,activations):
        self.rnkey = random.PRNGKey(1)
        rnkeys=random.split(self.rnkey,len(sizes))
        self.params=[]
        assert(len(activations.size)==len(sizes)-1)
        self.activations=[self.identity] 
        for z in activations:
            if z=='relu':
                self.activations.append(self.relu)
            elif z=='sigmoid':
                self.activations.append(self.sigmoid)
            if z=='identity':
                self.activations.append(self.identity)
        self.scale=0.01
        for n_prev,n_cur,key in zip(sizes[:-1], sizes[1:], rnkeys):
            x,y=random.split(key)
            self.params.append((self.scale*random.normal(x,(n_cur,n_prev)),self.scale*random.normal(y,(n_cur,))))

    def relu(self,X):
        return np.maximum(0,X)

    def sigmoid(self,X):
        return 1/(1+np.exp(-X))

    def identity(self,X):
        return X

    def forward_pass(self,params,X):
        ind=0
        activations = X
        for W,b in params:
            activations=self.activations[ind](np.matmul(activations,W.T)+b)
            ind+=1
        return activations

    def mse_loss(self, params, X, y):
        """ Compute the multi-class cross-entropy loss """
        preds = self.forward_pass(params, X)
        return np.sum((preds-y)**2)/preds.shape[0]

    def fit(self, X, y, batch_size=5, n_iter=10000, lr=0.001, lr_type='constant'):
        X=np.array(X).astype(np.float32)
        y=np.array(y).reshape(-1,1).astype(np.float32)

        m,n=X.shape

        opt_init, opt_update, get_params = optimizers.adam(lr)
        opt_state = opt_init(self.params)

        epochs=ceil(n_iter/floor(m/batch_size))

        for i in range(epochs):

            for j in range(0,m,batch_size):

                X_batch=X[j:j+batch_size]
                y_batch=y[j:j+batch_size]

                cur_loss, grads = value_and_grad(self.mse_loss)(self.params,X_batch,y_batch)
                opt_state = opt_update(0, grads, opt_state)
                self.params=get_params(opt_state)

                # print(cur_loss)
                
            
            print("Epoch: ",i,end=" ")
            print("cost: ",self.mse_loss(self.params,X,y))

        # print('Training loss %.6f' % self.loss(theta,X,y))

    def predict(self, X):
        m=X.shape[0]
        preds = self.forward_pass(self.params, X)
        return pd.Series(preds)