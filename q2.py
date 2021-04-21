import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from logisticRegression.logisticRegression import LogisticRegression
from metrics import *
from sklearn.datasets import load_breast_cancer
from sklearn import preprocessing
from sklearn.model_selection import KFold
import operator
from collections import OrderedDict
import pickle
import os

np.random.seed(42)

def l1(load=False):

	if not load:

		X,y=load_breast_cancer(return_X_y=True,as_frame=True)
		cols=X.columns
		X = preprocessing.StandardScaler().fit_transform(X)
		Xdf = pd.DataFrame(X,columns=cols)

		in_split = 3
		out_split = 4

		cv_outer = KFold(n_splits=out_split, shuffle=True, random_state=1)
		outer_results = list()

		# X=X[:60]
		# y=y[:60]

		# print(X.shape)
		# print(y.shape)

		ind = 0

		models_st=[]

		for train_ix, test_ix in cv_outer.split(X):
			ind += 1
			# split data
			X_train, X_test = X[train_ix, :], X[test_ix, :]
			y_train, y_test = y[train_ix], y[test_ix]
			# configure the cross-validation procedure
			cv_inner = KFold(n_splits=in_split, shuffle=True, random_state=1)
			# define the model
			model = LogisticRegression(fit_intercept=True)
			# define search space
			space = np.array([10**x for x in range(-3,3)])
			res={}
			for penalty in space:
				acc = []
				for train_ix2, test_ix2 in cv_inner.split(X_train):
					X_train2, X_test2 = X[train_ix2, :], X[test_ix2, :]
					y_train2, y_test2 = y[train_ix2], y[test_ix2]
					LR = LogisticRegression(fit_intercept=True,l1_coef=penalty)
					LR.fit_autograd(X_train2, y_train2, n_iter=1000) # here you can use fit_non_vectorised / fit_autograd methods
					y_hat2 = LR.predict(X_test2)
					acc.append(accuracy(y_hat2, y_test2))
				res[LR] = np.mean(acc)
			
			# for key,val in res.items():
			# 	print("l1_penalty: {}, acc: {}".format(key.l1_coef,val))

			best_model = max(res, key=res.get)
			
			models_st.append(best_model)

			# evaluate model on the hold out dataset
			yhat3 = best_model.predict(X_test)
			# evaluate the model
			acc = accuracy(y_test, yhat3)
			# store the result
			outer_results.append(acc)
			# report progress
			print('Fold=%d, acc=%.8f, best_l1_penalty=%.6f' % (ind, acc, best_model.l1_coef))
		# summarize the estimated performance of the model
		print("Overall Estimated Model Performance")
		print('Accuracy: %.5f (%.5f)' % (np.mean(outer_results), np.std(outer_results)))


		a_file = open("q2_data.pkl", "wb")
		pickle.dump((cols,models_st), a_file)
		a_file.close()

	a_file = open("q2_data.pkl", "rb")
	cols,models_st = pickle.load(a_file)
	a_file.close()

	# Feature Importance
	for model in models_st[:1]:

		print("Top 5 important features")

		theta = np.squeeze(model.coef_)[1:]
		res = dict()
		for i,val in enumerate(theta):
			res[cols[i]] = theta[i]
		
		res=OrderedDict(sorted(res.items(),key=lambda x:np.abs(x[1]),reverse=True))

		k = 5
		ind = 0
		for key,val in res.items():
			print("Feature: {}, coefficient: {}".format(key,val))
			ind += 1
			if(ind==k):
				break



def l2(load=False):

	if not load:

		X,y=load_breast_cancer(return_X_y=True,as_frame=True)
		cols=X.columns
		X = preprocessing.StandardScaler().fit_transform(X)
		Xdf = pd.DataFrame(X,columns=cols)

		in_split = 3
		out_split = 4

		cv_outer = KFold(n_splits=out_split, shuffle=True, random_state=1)
		outer_results = list()

		# X=X[:60]
		# y=y[:60]

		# print(X.shape)
		# print(y.shape)

		ind = 0

		models_st=[]

		for train_ix, test_ix in cv_outer.split(X):
			ind += 1
			# split data
			X_train, X_test = X[train_ix, :], X[test_ix, :]
			y_train, y_test = y[train_ix], y[test_ix]
			# configure the cross-validation procedure
			cv_inner = KFold(n_splits=in_split, shuffle=True, random_state=1)
			# define the model
			model = LogisticRegression(fit_intercept=True)
			# define search space
			space = np.array([10**x for x in range(-3,3)])
			res={}
			for penalty in space:
				acc = []
				for train_ix2, test_ix2 in cv_inner.split(X_train):
					X_train2, X_test2 = X[train_ix2, :], X[test_ix2, :]
					y_train2, y_test2 = y[train_ix2], y[test_ix2]
					LR = LogisticRegression(fit_intercept=True,l2_coef=penalty)
					LR.fit_autograd(X_train2, y_train2, n_iter=1000) # here you can use fit_non_vectorised / fit_autograd methods
					y_hat2 = LR.predict(X_test2)
					acc.append(accuracy(y_hat2, y_test2))
				res[LR] = np.mean(acc)
			
			# for key,val in res.items():
			# 	print("l1_penalty: {}, acc: {}".format(key.l1_coef,val))

			best_model = max(res, key=res.get)
			
			models_st.append(best_model)

			# evaluate model on the hold out dataset
			yhat3 = best_model.predict(X_test)
			# evaluate the model
			acc = accuracy(y_test, yhat3)
			# store the result
			outer_results.append(acc)
			# report progress
			print('Fold=%d, acc=%.8f, best_l2_penalty=%.6f' % (ind, acc, best_model.l2_coef))
		# summarize the estimated performance of the model
		print("Overall Estimated Model Performance")
		print('Accuracy: %.5f (%.5f)' % (np.mean(outer_results), np.std(outer_results)))

l1(False)
l2()