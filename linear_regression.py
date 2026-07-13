import math
import numpy as np


def compute_Phi(x,p):
    '''
        Input:
            x : a vector of samples in one dimensional space, a numpy vector of shape (n,).
                Here n is the number of samples.
            p : the number of polynomials/features
        Output:
            Phi: the design/feature matrix of x, a numpy array of shape (n,p).
    '''

    
    Phi = np.empty((len(x), p)) # empty 2D array with shape (n, p), rows are samples and columns are features

    for i in range(p):
        Phi[:, i] = x ** i # replace each column with x to the respective power

    return Phi 


def compute_yhat(Phi, w):
    '''
        Input:
            Phi: the feature matrix of all data instance, a float numpy array of shape (n,p). 
            w: the weights parameter of the linear model, a float numpy array of shape (p,). 
        Output:
            yhat: the logit value (predicted value) of all instances, a float numpy array of shape (n,)
    '''


    yhat = Phi @ w # dot product multiplies each feature with its weight and sums, resulting in vector yhat


    return yhat

def compute_L(yhat,y):
    '''
        Input:
            yhat: the predicted sample labels, a numpy vector of shape (n,).
            y:  the sample labels, a numpy vector of shape (n,).
        Output:
            L: the loss value of linear regression, a float scalar.
    '''


    L = np.mean((yhat - y) ** 2) / 2


    return L 



def compute_dL_dw(y, yhat, Phi):
    '''
        Input:
            Phi: the feature matrix of all data instances, a float numpy array of shape (n,p). 
               Here p is the number of features/dimensions.
            y: the sample labels, a numpy vector of shape (n,).
            yhat: the predicted sample labels, a numpy vector of shape (n,).
        Output:
            dL_dw: the gradients of the loss function L with respect to the weights w, a numpy float array of shape (p,). 

    '''


    dL_dw = (Phi.T @ (yhat - y)) / len(y) # computes gradient vector for all samples

    return dL_dw


def update_w(w, dL_dw, alpha = 0.001):
    '''
        Input:
            w: the current value of the weight vector, a numpy float array of shape (p,).
            dL_dw: the gradient of the loss function w.r.t. the weight vector, a numpy float array of shape (p,). 
            alpha: the step-size parameter of gradient descent, a float scalar.
        Output:
            w: the updated weight vector, a numpy float array of shape (p,).
    '''

    w = w - (alpha * dL_dw) # steps weights by a factor of alpha in opposite direction of gradient

    return w


def train(X, Y, alpha=0.001, n_epoch=100):
    '''
        Input:
            X: the feature matrix of training instances, a float numpy array of shape (n, p). Here n is the number of data instance in the training set, p is the number of features/dimensions.
            Y: the labels of training instance, a numpy integer array of shape (n,). 
            alpha: the step-size parameter of gradient descent, a float scalar.
            n_epoch: the number of passes to go through the training set, an integer scalar.
        Output:
            w: the weight vector trained on the training set, a numpy float array of shape (p,). 
    '''

    # initialize weights as 0
    w = np.array(np.zeros(X.shape[1])).T

    for _ in range(n_epoch):

        # Back propagation: compute local gradients 
        yhat = compute_yhat(X, w)
        dL_dw = compute_dL_dw(Y, yhat, X)

            
        # update the parameters w
        w = update_w(w, dL_dw, alpha)

    return w


