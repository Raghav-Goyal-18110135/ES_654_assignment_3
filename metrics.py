import numpy as np
import pandas as pd
from math import sqrt


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

def precision(y_hat, y, cls):
    """
    Function to calculate the precision
    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    > cls: The class chosen
    Output:
    > Returns the precision as float
    """
    assert(y_hat.size == y.size)
    ground=y.to_numpy(copy=True)
    pred=y_hat.to_numpy(copy=True)
    n=ground.size
    num=0
    deno=0
    for i in range(ground.size):
        if pred[i]==cls:
            deno+=1
            if ground[i]==cls:
                num+=1
    return (num/deno)


def recall(y_hat, y, cls):
    """
    Function to calculate the recall
    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    > cls: The class chosen
    Output:
    > Returns the recall as float
    """
    assert(y_hat.size == y.size)
    ground=y.to_numpy(copy=True)
    pred=y_hat.to_numpy(copy=True)
    num=0
    deno=0
    for i in range(ground.size):
        if ground[i]==cls:
            deno+=1
            if pred[i]==cls:
                num+=1
    return (num/deno)

def rmse(y_hat, y):
    """
    Function to calculate the root-mean-squared-error(rmse)
    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the rmse as float
    """

    y=pd.Series(y)
    y_hat=pd.Series(y_hat)

    assert(y_hat.size == y.size)
    ground=y.to_numpy(copy=True)
    pred=y_hat.to_numpy(copy=True)
    ans=0
    for i in range(ground.size):
        ans+=(pred[i]-ground[i])**2
    ans/=ground.size
    return sqrt(ans)

def mae(y_hat, y):
    """
    Function to calculate the mean-absolute-error(mae)
    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the mae as float
    """
    assert(y_hat.size == y.size)
    ground=y.to_numpy(copy=True)
    pred=y_hat.to_numpy(copy=True)
    ans=0
    for i in range(ground.size):
        ans+=abs(pred[i]-ground[i])
    ans/=ground.size
    return ans