# imports
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from scipy.linalg import eigh
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import warnings
import copy
from statistics import mean
import os
warnings.filterwarnings('ignore')

spam = pd.read_csv("https://web.stanford.edu/~hastie/ElemStatLearn/datasets/spam.data", sep=" ", na_values="?", header=None)
spam = spam.dropna()

trainTestInd = pd.read_csv("https://web.stanford.edu/~hastie/ElemStatLearn/datasets/spam.traintest", sep= " ", header=None)
spam = np.hstack((spam, trainTestInd))

# seperate train and test using website indicator
trainSpam = spam[spam[:,-1] == 0]
testSpam = spam[spam[:,-1] == 1]

# grab X values
X_train = trainSpam[:, 0:57]
X_test = testSpam[:, 0:57]

# grab y values, replace 0's and 1's with -1's and 1's
y_train = (trainSpam[:,-2])
y_train = y_train.reshape((len(y_train), 1))
y_train[y_train == 0] = -1
y_test = testSpam[:,-2]
y_test = y_test.reshape((len(y_test), 1))
y_test[y_test == 0] = -1


# standardizing X by subtracting the mean of the predictors and dividing by their standard deviation
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

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

# set constants
rho = 2
epsilon = 0.001
lamb = 1

# calculate starting step size
n = len(X_train)
eq = (1/n * X_train.T.dot(X_train))
eigenVals = eigh(eq)[0]
initialStepSize = 1 / (max(eigenVals) + lamb)

# train algorithm
trainBetas, trainObjs = myrhologistic(X_train, y_train, initialStepSize, epsilon, rho, lamb)

def calcMisclassificationError(X, y, betas):
    """
    Calculates the misclassification error for a binary response variable
    Inputs:
        - X: matrix of X values
        - y: vector of associated outcomes, binary
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
            if(prediction > 0): # maps the 0 and 1 values to -1 and 1
                prediction = 1
            else:
                prediction = -1
            if(prediction != y[i]):
                incorrect +=1
        misclassifications[b,:] = (incorrect / X.shape[0])
    return misclassifications

# calculate misclassification error
trainMisclassification = calcMisclassificationError(X_train, y_train, trainBetas)

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
            #modelMisclassification = calcMisclassificationError(xRow, yRow, modelBetas[-1])
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

# leave one out cross validation
lambdas = [0.0001, 0.001, 0.01, 0.1, 1]
lOOLamb = leaveOneOutCrossValidation(X_train, y_train, lambdas, 0.001, 2)
print(lOOLamb)
