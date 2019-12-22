"""
Do not change the input and output format.
If our script cannot run your code or the format is improper, your code will not be graded.

The only functions you need to implement in this template is linear_regression_noreg, linear_regression_invertible，regularized_linear_regression,
tune_lambda, test_error and mapping_data.
"""

import numpy as np
import pandas as pd

###### Q1.1 ######
def mean_absolute_error(w, X, y):
    """
    Compute the mean absolute error on test set given X, y, and model parameter w.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing test feature.
    - y: A numpy array of shape (num_samples, ) containing test label
    - w: a numpy array of shape (D, )
    Returns:
    - err: the mean absolute error
    """
    #####################################################
    # TODO 1: Fill in your code here #

    y_pred = np.sum(np.multiply(X, np.transpose(w)),axis = 1)
    if X.shape[0] != 0:
        err = np.sum(np.absolute(np.transpose(y_pred)-y))/X.shape[0]
    else:
        return 0.0
    #####################################################
    # err = None
    return err

###### Q1.2 ######
def linear_regression_noreg(X, y):
  """
  Compute the weight parameter given X and y.
  Inputs:
  - X: A numpy array of shape (num_samples, D) containing feature.
  - y: A numpy array of shape (num_samples, ) containing label
  Returns:
  - w: a numpy array of shape (D, )
  """
  #####################################################
  #	TODO 2: Fill in your code here #
  w = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(X),X)), np.transpose(X)),y)
  #####################################################
  # w = None
  return w

###### Q1.3 ######
def linear_regression_invertible(X, y):
    """
    Compute the weight parameter given X and y.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    Returns:
    - w: a numpy array of shape (D, )
    """
    #####################################################
    # TODO 3: Fill in your code here #
    XX = np.matmul(np.transpose(X),X)
    e_val, e_vec = np.linalg.eig(XX)
    add = 0.1*np.ones(X.shape[1])
    while np.amin(np.absolute(e_val)) < np.power(10.0,-5):
        e_val += add
    XX_vir = np.matmul(np.matmul(e_vec,np.diag(e_val)),np.linalg.inv(e_vec))
    w = np.matmul(np.matmul(np.linalg.inv(XX_vir), np.transpose(X)),y)
    #####################################################
    # w = None
    return w


###### Q1.4 ######
def regularized_linear_regression(X, y, lambd):
    """
    Compute the weight parameter given X, y and lambda.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    - lambd: a float number containing regularization strength
    Returns:
    - w: a numpy array of shape (D, )
    """
    #####################################################
    # TODO 4: Fill in your code here #
    FF = np.matmul(np.transpose(X),X)
    # print(FF)
    e_val, e_vec = np.linalg.eig(FF)
    add = lambd*np.ones(X.shape[1])

    e_val += add
    FF_vir = np.matmul(np.matmul(e_vec,np.diag(e_val)),np.linalg.inv(e_vec))
    w = np.matmul(np.matmul(np.linalg.inv(FF_vir), np.transpose(X)),y)
    #####################################################
    # w = None
    return w

###### Q1.5 ######
def tune_lambda(Xtrain, ytrain, Xval, yval):
    """
    Find the best lambda value.
    Inputs:
    - Xtrain: A numpy array of shape (num_training_samples, D) containing training feature.
    - ytrain: A numpy array of shape (num_training_samples, ) containing training label
    - Xval: A numpy array of shape (num_val_samples, D) containing validation feature.
    - yval: A numpy array of shape (num_val_samples, ) containing validation label
    Returns:
    - bestlambda: the best lambda you find in lambds
    """
    #####################################################
    # TODO 5: Fill in your code here #
    bestlambda = np.power(10.0,-19)
    mae_min = 10
    for i in range(-19, 20):
        w_curr = regularized_linear_regression(Xtrain, ytrain, np.power(10.0,i))
        mae_curr = mean_absolute_error(w_curr, Xval, yval)
        if mae_min >= mae_curr:
            mae_min = mae_curr
            bestlambda = np.power(10.0,i)
    #####################################################
    # bestlambda = None
    return bestlambda


###### Q1.6 ######
def mapping_data(X, power):
    """
    Mapping the data.
    Inputs:
    - X: A numpy array of shape (num_training_samples, D) containing training feature.
    - power: A integer that indicate the power in polynomial regression
    Returns:
    - X: mapped_X, You can manully calculate the size of X based on the power and original size of X
    """
    #####################################################
    # TODO 6: Fill in your code here #
    X_temp = X
    if power == 1:
        return X
    else:
        for i in range(2,power+1):
            X_curr_pow = np.power(X_temp,i)
            for j in range(X_curr_pow.shape[1]):
                X = np.insert(X, X.shape[1], np.transpose(X_curr_pow)[j], axis=1)
            print(X.shape)
    #####################################################

    return X
