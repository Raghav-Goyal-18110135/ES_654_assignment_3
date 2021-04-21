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
from autograd import jacobian
np.seterr(divide='ignore', invalid='ignore')


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
        self.params=[(rs.randn(insize, outsize) * scale,   # weight matrix
             rs.randn(outsize) * scale)           # bias vector
            for insize, outsize in zip(sizes[:-1], sizes[1:])]

    def relu(self,X):
        return np.maximum(0,X)

    def sigmoid(self,X):
        return 1/(1+np.exp(-X))

    def identity(self,X):
        return X

    def forward_pass(self, params, inputs):
        ind=0
        for W, b in params[:]:
            outputs = np.dot(inputs, W) + b
            inputs = self.activations[ind](outputs)
            ind+=1
        # no activation on the last layer
        # W, b = params[-1]
        # return np.dot(inputs, W) + b
        return outputs

    def objective(self, params, _):

        # print(X.shape)
        # print(y.shape)
        """ Compute the multi-class cross-entropy loss """
        preds = self.forward_pass(params, self.X_batch)
        # print(preds.shape)
        preds=np.exp(preds)
        predsum=np.sum(preds,axis=1,keepdims=True)
        preds/=predsum
        # print(np.sum(preds,axis=1))
        res = 0
        self.y_batch=self.y_batch.squeeze()
        for i in range(preds.shape[0]):
            res -= np.log10(preds[i][int(self.y_batch[i])])
        res=res/preds.shape[0]
        # print(res)
        return res

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

        objective_grad = grad(self.objective)

        for i in range(epochs):
            for j in range(0,m,batch_size):

                self.X_batch=X[j:j+batch_size]
                self.y_batch=y[j:j+batch_size]

                step_size = lr

                self.params = adam(objective_grad, self.params, step_size=step_size,
                            num_iters=1)


            self.X_batch=X
            self.y_batch=y

            # print("Epoch: ",i,end=" ")
            # print("cost: ",self.objective(self.params,0))

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



#############################################################################################################
#############################################################################################################

class mlp_regressor():

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
        scale=0.1
        rs=npr.RandomState(0)
        self.params=[(rs.randn(insize, outsize) * scale,   # weight matrix
             rs.randn(outsize) * scale)           # bias vector
            for insize, outsize in zip(sizes[:-1], sizes[1:])]

    def relu(self,X):
        return np.maximum(0,X)

    def sigmoid(self,X):
        return 1/(1+np.exp(-X))

    def identity(self,X):
        return X

    def forward_pass(self, params, inputs):
        ind=0
        for W, b in params[:]:
            outputs = np.dot(inputs, W) + b
            inputs = self.activations[ind](outputs)
            ind+=1
        # no activation on the last layer
        # W, b = params[-1]
        # return np.dot(inputs, W) + b
        return outputs

    def objective(self, params, _):

        # print(X.shape)
        # print(y.shape)
        """ Compute the multi-class cross-entropy loss """
        preds = self.forward_pass(params, self.X_batch)
        assert(preds.shape==self.y_batch.shape)
        err=preds-self.y_batch
        return np.mean(err**2)

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

        objective_grad = grad(self.objective)

        for i in range(epochs):
            for j in range(0,m,batch_size):

                self.X_batch=X[j:j+batch_size]
                self.y_batch=y[j:j+batch_size]

                step_size = lr

                self.params = adam(objective_grad, self.params, step_size=step_size,
                            num_iters=1, b1=0.9, b2=0.999)


            self.X_batch=X
            self.y_batch=y

            # print("Epoch: ",i,end=" ")
            # print("cost: ",self.objective(self.params,0))

        # print('Training loss %.6f' % self.loss(theta,X,y))

    def predict(self, X):
        res=self.forward_pass(self.params,X)
        # print(res.shape)
        return pd.Series(res.squeeze())