import autograd.numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from math import sqrt,floor,ceil
# Import Autograd modules here
# import jax.numpy as jnp
# from jax import grad, jit, vmap
from autograd import grad


class kclassLogisticRegression():

    def __init__(self, num_classes, fit_intercept=True,l1_coef=0,l2_coef=0):
        '''
        :param fit_intercept: Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).
        '''
        self.fit_intercept = fit_intercept
        self.coef_ = None #Replace with numpy array or pandas series of coefficients learned using using the fit methods
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.num_classes = num_classes

    def predicition(self,theta,X):
        assert(X.shape[1]==theta.shape[0])
        mat=np.exp(np.matmul(X,theta))
        mat_sum=np.sum(mat,axis=1,keepdims=True)
        mat/=mat_sum
        return mat

    def loss(self,theta,x,y):
        assert(x.shape[0]==y.shape[0])
        pred=self.predicition(theta,x)
        assert(pred.shape==(x.shape[0],self.num_classes))
        res = 0
        y=y.squeeze()
        for i in range(x.shape[0]):
            res -= np.log10(pred[i][int(y[i])])
        res=res/x.shape[0]
        # print(res)
        return res

    def fit_non_vectorised(self, X, y, batch_size=5, n_iter=1000, lr=0.01, lr_type='constant'):

        # print(X)
        # print(y)

        self.y=y

        X=np.array(X).astype(np.float64)
        y=np.array(y).reshape(-1,1).astype(np.float64)
        m=X.shape[0]
        n=X.shape[1]
        if self.fit_intercept:
            X=np.concatenate((np.ones(m).reshape(-1,1),X),axis=1)
            # print(X)
            n+=1
        num_classes=self.num_classes
        theta=np.zeros((n,self.num_classes))
        epochs=ceil(n_iter/floor(m/batch_size))
        
        for i in range(epochs):
            for j in range(0,m,batch_size):
                if lr_type=='constant':
                    lrate=lr
                elif lr_type=='inverse':
                    it=floor(m/batch_size)*i+j+1
                    lrate=lr/(it)
                X_batch=X[j:j+batch_size]
                y_batch=y[j:j+batch_size]
                preds=self.predicition(theta,X_batch)

                indicator = np.zeros((int(y_batch.size), num_classes))
                indicator[np.arange(int(y_batch.size)).astype(int),np.array(y_batch).squeeze().astype(int)] = 1

                assert(indicator.shape==preds.shape)
                preds=indicator-preds

                for z in range(num_classes):
                    xx=preds[:,z].squeeze()
                    vec=np.zeros(X_batch.shape[1])
                    for k in range(X_batch.shape[0]):
                        vec+=xx[k]*X_batch[k,:]
                        # print(a)
                        # print(b)
                    assert(theta[:,z].shape==vec.shape)
                    theta[:,z]+=lrate*vec
                
                # print("Iteration: ",floor(m/batch_size)*i+j+1,end=" ")
                # print("cost: ",self.loss(theta,X_batch,y_batch))

            # print("Epoch: ",i,end=" ")
            # print("cost: ",self.loss(theta,X,y))

        self.coef_=theta

    def fit_autograd(self, X, y, batch_size=5, n_iter=10000, lr=0.01, lr_type='constant'):

        self.y=y
                
        # print(X)
        # print(y)

        X=np.array(X).astype(np.float64)
        y=np.array(y).reshape(-1,1).astype(np.float64)
        m=X.shape[0]
        n=X.shape[1]

        # print(X.shape)
        # print(y.shape)

        if self.fit_intercept:
            X=np.concatenate((np.ones(m).reshape(-1,1),X),axis=1)
            # print(X)
            n+=1
        theta=np.zeros((n,self.num_classes))
        epochs=ceil(n_iter/floor(m/batch_size))

        gradfn=grad(self.loss)


        for i in range(epochs):
            # indices=np.random.permutation(m)
            # X=X[indices]
            # y=y[indices]
            # print("Epoch: ",i)
            # print(X)
            # print(y)

            for j in range(0,m,batch_size):
                X_batch=X[j:j+batch_size]
                y_batch=y[j:j+batch_size]
                preds=self.predicition(theta,X_batch)
                if lr_type=='constant':
                        lrate=lr
                elif lr_type=='inverse':
                    it=floor(m/batch_size)*i+j+1
                    lrate=lr/(it)
                gradient=gradfn(theta,X_batch,y_batch)
                # print(gradient.shape)
                theta-=lrate*gradient
                
                # print("Iteration: ",floor(m/batch_size)*i+j+1,end=" ")
                # k=self.loss(theta,X_batch,y_batch)
                # print(k,"**")
                # print("cost: ",self.loss(theta,X,y))
            
            # print("Epoch: ",i,end=" ")
            # print("cost: ",self.loss(theta,X,y))

        # print('Training loss %.6f' % self.loss(theta,X,y))

        self.coef_=theta

    def predict(self, X):

        m=X.shape[0]
        if self.fit_intercept:
            X=np.concatenate((np.ones(m).reshape(-1,1),X),axis=1)
        pred=self.predicition(self.coef_,X)
        res=np.argmax(pred,axis=1)
        # print(pd.DataFrame({0:pred,1:res,2:np.array(self.y).squeeze()}))
        return pd.Series(res)


    def predict_proba(self, X):

        m=X.shape[0]
        if self.fit_intercept:
            X=np.concatenate((np.ones(m).reshape(-1,1),X),axis=1)

        pred=self.predicition(self.coef_,X)

        return pd.Series(pred)