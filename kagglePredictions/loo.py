# imports
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from scipy.linalg import eigh
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from statistics import mean
from collections import Counter
import warnings
import copy
import os
warnings.filterwarnings('ignore')

def leaveOneOutCrossValidation(X, y, lambdas, epsilon, rho):
    optimizedLamb = 0
    misclassificationRate = 1
    for lamb in lambdas:
        print("LAMBDA :", lamb)
        scores = []
        for i in range(len(X)):
            xRow = X[i]
            xRest = np.delete(copy.copy(X), i, 0)
            yRow = y[i][0]
            yRest = np.delete(copy.copy(y), i, 0)
            n = len(xRest)
            eq = (1/n * xRest.T.dot(xRest))
            eigenVals = eigh(eq)[0]
            initialStepSize = 1 / (max(eigenVals) + lamb)
            modelBetas, modelObjs = myrhologistic(xRest, yRest, initialStepSize, epsilon, rho, lamb)
            prediction = np.sum(np.dot(xRow, modelBetas[-1]))
            score = 1
            if(prediction > 0): # maps the 0 and 1 values to -1 and 1
                prediction = 1
            else:
                prediction = -1
            if(prediction == yRow):
                score = 0
            scores.append(score)
        if(mean((scores)) < misclassificationRate):
            misclassificationRate = mean((scores))
            optimizedLamb = lamb
    return optimizedLamb

def myrhologistic(X, y, stepSize, targetAccuracy, rho, lamb):
    """
    Implements accelerated gradient descent algorithm with backtracking
    for l-2 regularized binary logistic regression with p-logistic loss
    Inputs:
        - X: matrix of X values
        - y: vector of associated outcomes
        - stepSize: initial step size
        - targetAccuracy: target accuracy value for algorithm
        - rho: factor for logistic loss
        - lambda: scalar multiplicative factor for regularization penalty
    Outputs:
        - betas: vector of improved betas after final iteration
        - objs: vector of objective values for each iteration
    """
    beta = np.zeros((X.shape[1], 1))
    theta = np.zeros((X.shape[1], 1))
    objs = [computeobj(X, y, beta, rho, lamb)]
    betas = [beta]
    grad = computegrad(X, y, beta, rho, lamb)
    t = 0
    while (np.linalg.norm(grad) > targetAccuracy):
        stepSize = backtracking(X, y, beta, stepSize, rho, lamb)
        betaOld = copy.copy(beta)
        beta = theta - stepSize * computegrad(X, y, theta, rho, lamb)
        theta = beta + (t/(t+3))*(beta - betaOld)
        betas.append(beta) # saves current beta values
        obj = computeobj(X, y, beta, rho, lamb)
        objs.append(obj) # saves current objective value
        grad = computegrad(X, y, beta, rho, lamb)
        t += 1
    return np.array(betas), np.array(objs)

def computegrad(X, y, beta, rho, lamb):
    """
    Computes the gradient for the fast gradient algorithm
    Inputs:
        - X: matrix of X values
        - y: vector of associated outcomes
        - beta: vector of beta constants
        - lambda: scalar multiplicative factor for regularization penalty (optional, defaults to 0.05)
    Outputs:
        - vector gradient for passed in parameters
    """
    n = len(X)
    summation = 0
    for i in range(0, X.shape[0]):
        xi = X[i,:]
        yi = y[i]
        expTerm = ((np.exp(-rho * yi * (xi.T).dot(beta))) / (1 + (np.exp(-rho * yi * (xi.T).dot(beta)))))
        summation += yi * xi * expTerm
    summation = summation.reshape(summation.shape[0], 1)
    return ((-1/n) * summation + (2*lamb*beta).reshape(summation.shape[0],1))

def computeobj(X, y, beta, rho, lamb):
    """
    Computes the objective for ridge regression problem
    Inputs:
        - X: matrix of X values
        - y: vector of associated outcomes
        - beta: vector of beta constants
        - lambda: scalar multiplicative factor for regularization penalty (optional, defaults to 0.05)
    Outputs:
        - objective for passed in parameters
    """
    n = len(X)
    summation = 0
    for i in range(0, n):
        xi = X[i,:]
        yi = y[i]
        x = -rho*yi*xi.T.dot(beta)
        logTerm = np.log(1 + np.exp(x))
#         if(x > 0): # avoids over or under flow errors
#             a = x + 1
#             logTerm = a + np.log(np.exp(-a) + np.exp(x-a))
#         else:
#             logTerm = np.log(1 + np.exp(x))
        summation = summation + logTerm
    return ((1/(n * rho)) * summation + (lamb * np.sum(beta**2)))[0]

def backtracking(X, y, beta, eta, rho, lamb, alpha = 0.5, gamma=0.8):
    """
    Implements backtracking rule
    Inputs:
        - X: matrix of X values
        - y: vector of associated outcomes
        - beta: vector of beta constants
        - eta: initial step size
        - alpha: constant used to define sufficinet decrease condition
        - gamma: constant to scale step size by until condition met
    Outputs:
        - step size
    """
    grad = computegrad(X, y, beta, rho, lamb)  # calculates the gradient at current beta
    conditionMet = False # tracks when we find the backtracked step size
    while not conditionMet:
        if computeobj(X, y, beta - eta*grad, rho, lamb) < (computeobj(X, y, beta, rho, lamb) - alpha*eta*np.linalg.norm(grad)**2):
            conditionMet = True
        else:
            eta = eta * gamma
    return eta

x_train_unfiltered = np.load('train_features.npy')
y_train_unfiltered = np.load('train_labels.npy')
x_test_unfiltered = np.load('val_features.npy')
y_test_unfiltered = np.load('val_labels.npy')

kaggleTest = np.load('test_features.npy')

# filter testing data to two classes: 4, 7
x_train = np.zeros((1000, x_train_unfiltered.shape[1]))
y_train = np.zeros((1000, 1))
index = 0
for i in range(y_train_unfiltered.shape[0]):
    if(y_train_unfiltered[i] == 4 or y_train_unfiltered[i] == 7):
        x_train[index, :] = x_train_unfiltered[i].T
        y_train[index, :] = y_train_unfiltered[i]
        index += 1

y_train[y_train == 4] = -1
y_train[y_train == 7] = 1

# filter testing data to two classes: 4, 7
x_test = np.zeros((200, x_test_unfiltered.shape[1]))
y_test = np.zeros((200, 1))
index = 0

for i in range(y_test_unfiltered.shape[0]):
    if(y_test_unfiltered[i] == 4 or y_test_unfiltered[i] == 7):
        x_test[index, :] = x_test_unfiltered[i].T
        y_test[index, :] = y_test_unfiltered[i]
        index += 1

# reset y data to -1 and 1 to fit machine learning standards
y_test[y_test == 4] = -1
y_test[y_test == 7] = 1

# standardize x values
xScaler = preprocessing.StandardScaler()
X_train = xScaler.fit_transform(x_train)
X_test = xScaler.transform(x_test)

# set constants
rho = 2
epsilon = .001
lamb = 1

# calculate starting step size for training data
n = len(X_train)
eq = (1/n * X_train.T.dot(X_train))
eigenVals = eigh(eq)[0]
kaggleTrainInitialStepSize = 1 / (max(eigenVals) + lamb)


def calcKaggleMisclassificationError(X, y, betas):
    """
    Calculates the misclassification error for a 0-9 response variable
    Inputs:
        - X: matrix of X values
        - y: vector of associated outcomes, 0-9
        - betas: coefficients that correspond to X values
    Outputs:
        - misclassifications: ratio of incorrect values / total values for all betas
    """
    misclassifications = np.zeros((betas.shape[0], 1))
    for b in range(betas.shape[0]):
        betaVals = betas[b]
        incorrect = 0
        for i in range(X.shape[0]):
            prediction = np.sum(np.dot(X[i], betaVals))
            if(prediction > 0):
                prediction = 1
            else:
                prediction = -1
            if(prediction != y[i]):
                incorrect +=1
        misclassifications[b,:] = (incorrect / X.shape[0])
    return misclassifications

lambdas = [0.0001, 0.001, 0.01, 0.1, 1]
lOOLamb = leaveOneOutCrossValidation(X_train, y_train, lambdas, 0.001, 2)
print("DONE")
print(lOOLamb)
