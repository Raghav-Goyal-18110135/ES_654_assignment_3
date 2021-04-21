import autograd.numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from math import sqrt,floor,ceil
# Import Autograd modules here
# import jax.numpy as jnp
# from jax import grad, jit, vmap
from autograd import grad


class LogisticRegression():

    def __init__(self, fit_intercept=True,l1_coef=0,l2_coef=0):
        '''
        :param fit_intercept: Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).
        '''
        self.fit_intercept = fit_intercept
        self.coef_ = None #Replace with numpy array or pandas series of coefficients learned using using the fit methods
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
    
    def sigmoid(self,z):
        return 1/(1+np.exp(-1*z))
        # return 0.5*(np.tanh(z) + 1)

    def predicition(self,theta,X):
        assert(X.shape[1]==theta.shape[0])
        return self.sigmoid(np.matmul(X,theta))

    def loss(self,theta,x,y):
        assert(x.shape[0]==y.shape[0])
        pred=self.predicition(theta,x)
        pred=np.squeeze(pred)
        y=np.squeeze(y)
        res=-np.sum(y.dot(np.log10(pred))+(1-y).dot(np.log10(1-pred)))
        res=res/x.shape[0]
        res+=self.l2_coef/(2*x.shape[0])*np.sum(np.square(theta))
        res+=self.l1_coef/(2*x.shape[0])*np.sum(np.abs(theta))
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
        theta=np.zeros(n).reshape(-1,1)
        epochs=ceil(n_iter/floor(m/batch_size))
        
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
                # print(theta.shape)
                # print(X_batch.shape)
                preds=self.predicition(theta,X_batch)
                for k in range(n):
                    if lr_type=='constant':
                        lrate=lr
                    elif lr_type=='inverse':
                        it=floor(m/batch_size)*i+j+1
                        lrate=lr/(it)
                    a=(preds-y_batch)
                    b=X_batch[:,[k]]
                    # print(a)
                    # print(b)
                    theta[k][0]-=lrate*np.sum(a*b)
                
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
        theta=np.zeros(n).reshape(-1,1)
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
            
        #     print("Epoch: ",i,end=" ")
        #     print("cost: ",self.loss(theta,X,y))

        # print('Training loss %.6f' % self.loss(theta,X,y))

        self.coef_=theta

    def predict(self, X):

        m=X.shape[0]
        if self.fit_intercept:
            X=np.concatenate((np.ones(m).reshape(-1,1),X),axis=1)
        pred=self.predicition(self.coef_,X)
        pred=np.squeeze(pred)
        res=np.squeeze(pred>=0.5).astype(int)
        # print(pd.DataFrame({0:pred,1:res,2:np.array(self.y).squeeze()}))
        return pd.Series(res)

    def predict_proba(self, X):

        m=X.shape[0]
        if self.fit_intercept:
            X=np.concatenate((np.ones(m).reshape(-1,1),X),axis=1)

        pred=self.predicition(self.coef_,X)
        pred=np.squeeze(pred)
        res=np.squeeze(pred).astype(float)
        # print(pd.DataFrame({0:pred,1:res,2:np.array(self.y).squeeze()}))
        return pd.Series(res)

    def plot_surface(self, X, y, f0, f1):

        Xorig=X
        X=np.array(X)[:,[f0,f1]]
        y=np.array(y)
        a = np.linspace(-5, 5, 1000)
        b = np.linspace(-5, 5, 1000)
        xx, yy = np.meshgrid(a,b)
        grid = np.c_[xx.ravel(), yy.ravel()]
        zero_extender = np.zeros((grid.shape[0],Xorig.shape[1]-grid.shape[1]))
        grid=np.concatenate((zero_extender,grid),axis=1)
        res=np.array(self.predict_proba(grid))
        probs = np.array(res.reshape(xx.shape))
        

        f, ax = plt.subplots(figsize=(8, 6))
        contour = ax.contourf(xx, yy, probs, 25, cmap="RdBu",vmin=0, vmax=1)
        ax_c = f.colorbar(contour)
        ax_c.set_label("$P(y = 1)$")
        ax_c.set_ticks([0, .25, .5, .75, 1])

        ax.scatter(X[:, 0], X[:, 1], c=y[:], s=50,
                cmap="RdBu", vmin=-.2, vmax=1.2,
                edgecolor="white", linewidth=1)

        ax.set(aspect="equal",
            xlim=(-5, 5), ylim=(-5, 5),
            xlabel=list(Xorig.columns)[f0]+" (normalised)", ylabel=list(Xorig.columns)[f1]+" (normalised)")

        plt.savefig("q1_d.png")
        plt.show()
