import math
import numpy as np


'''
    Notations:
            ---------- input data ----------------------
            p: the number of input features.
            x: the feature vector of a data instance, a float numpy array of shape (p,). 
            y: the label of a training instance, an integer scalar value. The values can be 0 or 1.

            ---------- model parameters ----------------------
            In this code, we use w and b instead of theta.
            w: the weights parameter of the logistic model, a float numpy array of shape (p,).
            b: the bias parameter of the logistic model, a float scalar.

            ---------- values ----------------------
            z: the logit value, a float scalar
            a: the activation value, a float scalar
            L: the cross entropy loss value, a float scalar.

            ---------- partial gradients ----------------------
            dL_da: the partial gradient of the loss function L w.r.t. the activation a, a float scalar value. It represents (d_L / d_a)
            da_dz: the partial gradient of the activation a w.r.t. the logit z, a float scalar value. It represents (d_a / d_z)
            dz_dw: the partial gradients of the logit z w.r.t. the weights w, a float numpy array of shape (p,). It represents (d_z / d_w)
                   The i-th element represents ( d_z / d_w[i])
            dz_db: the partial gradient of logit z w.r.t. the bias b, a float scalar. It represents (d_z / d_b).

            ---------- partial gradients of parameters ------------------
            dL_dw: the partial gradient of the loss function L w.r.t. the weight vector w, a float numpy array of shape (p,). 
                   The i-th element represents ( d_L / d_w[i])
            dL_db: the partial gradient of the loss function L w.r.t. the bias b, a float scalar. 

            ---------- training ----------------------
            alpha: the step-size parameter of gradient descent, a float scalar.
            n_epoch: the number of passes to go through the training dataset in the training process, an integer scalar.
'''

#-----------------------------------------------------------------
# Forward Pass 
#-----------------------------------------------------------------


def compute_z(x,w,b):
    '''
        Input:
            x: the feature vector of a data instance, a float numpy array of shape (p,). 
            w: the weights parameter of the logistic model, a float numpy array of shape (p,). 
            b: the bias value of the logistic model, a float scalar.
        Output:
            z: the logit value of the instance, a float scalar
    '''
    
    z = np.sum(x * w) + b
    
    return z


def compute_a(z):
    '''
        Input:
            z: the logit value of logistic regression, a float scalar.
        Output:
            a: the activation, a float scalar
    '''

    z = np.ravel(z)[0] # convert to scalar

    if z >= 0:
        a = 1 / (1 + math.exp(-z))
    else: # avoid overflow
        ez = math.exp(z)
        a = ez / (1 + ez)


    return a


def compute_L(a,y):
    '''
        Input:
            a: the activation of a training instance, a float scalar
            y: the label of a training instance, an integer scalar value. The values can be 0 or 1.
        Output:
            L: the loss value of logistic regression, a float scalar.
    '''

    a = np.ravel(a)[0] # convert to scalar

    if y == 1:
        if a == 0:
            L = 1e100 # avoid overflow
        else:
            L = -math.log(a)
    else:
        if a == 1:
            L = 1e100 # avoid overflow
        else:
            L = -math.log(1 - a)

    return L 


def forward(x,y,w,b):
    '''
        Input:
            x: the feature vector of a training instance, a float numpy array of shape (p,). Here p is the number of features/dimensions.
            y: the label of a training instance, an integer scalar value. The values can be 0 or 1.
            w: the weight vector, a float numpy array of shape (p,).
            b: the bias value, a float scalar.
        Output:
            z: linear logit of the instance, a float scalar
            a: activation, a float scalar
            L: the cross entropy loss on the training instance, a float scalar. 
    '''


    z = compute_z(x, w, b)
    a = compute_a(z)
    L = compute_L(a, y)
    
    return z, a, L 



#-----------------------------------------------------------------
# Compute Local Gradients
#-----------------------------------------------------------------


#--------------------------
def compute_dL_da(a, y):
    '''
        Input:
            a: the activation value, a float scalar
            y: the label of a training instance, an integer scalar value. The values can be 0 or 1.
        Output:
            dL_da: the local gradient of the loss function w.r.t. the activation, a float scalar value.
    '''


    if y == 0:
        if a == 1:
            dL_da = 1e100 # avoid overflow
        else:
            dL_da = 1 / (1-a)
    else:
        if a == 0:
            dL_da = -1e100 # avoid overflow
        else:
            dL_da = -1/a


    return dL_da 


 
#--------------------------
def compute_da_dz(a):
    '''
        Input:
            a: the activation value of the sigmoid function, a float scalar
        Output:
            da_dz: the local gradient of the activation w.r.t. the logit z, a float scalar value.
    '''


    da_dz = a * (1 - a)

    return da_dz 

#--------------------------
def compute_dz_dw(x):
    '''
        Compute partial gradients of the logit function z with respect to (w.r.t.) the weights w. 
        Input:
            x: the feature vector of a data instance, a float numpy array of shape (p,). 
               Here p is the number of features/dimensions.
        Output:
            dz_dw: the partial gradients of the logit z with respect to the weights w, a float numpy array of shape (p,). 
                   The i-th element represents ( d_z / d_w[i])
    '''


    dz_dw = x


    return dz_dw

 
#--------------------------
def compute_dz_db():
    '''
        Compute partial gradient of the logit function z with respect to (w.r.t.) the bias b. 
        Output:
            dz_db: the partial gradient of logit z with respect to the bias b, a float scalar. It represents (d_z / d_b).
    '''


    dz_db = 1


    return dz_db


#-----------------------------------------------------------------
# Back Propagation 
#-----------------------------------------------------------------

#--------------------------
def backward(x,y,a):
    ''' 
        Input:
            x: the feature vector of a data instance, a float numpy array of shape (p,). 
            y: the label of a training instance, an integer scalar value. The values can be 0 or 1.
            a: the activation, a float scalar
        Output:
            dL_da: the local gradient of the loss function w.r.t. the activation, a float scalar value.
            da_dz: the local gradient of the activation a w.r.t. the logit z, a float scalar value. It represents ( d_a / d_z )
            dz_dw: the partial gradient of logit z with respect to the weight vector, a float numpy array of shape (p,). 
                   The i-th element represents ( d_z / d_w[i])
            dz_db: the partial gradient of logit z with respect to the bias, a float scalar. It represents (d_z / d_b).
    '''


    dL_da = compute_dL_da(a, y)
    da_dz = compute_da_dz(a)
    dz_dw = compute_dz_dw(x)
    dz_db = compute_dz_db()

    return dL_da, da_dz, dz_dw, dz_db 



#--------------------------
def compute_dL_dw(dL_da, da_dz, dz_dw):
    '''
        Input:
            dL_da: the local gradient of the loss function w.r.t. the activation, a float scalar value.
            da_dz: the local gradient of the activation a w.r.t. the logit z, a float scalar value. It represents ( d_a / d_z )
            dz_dw: the partial gradient of logit z with respect to the weight vector, a float numpy array of shape (p,). 
                   The i-th element represents ( d_z / d_w[i])
        Output:
            dL_dw: the gradient of the loss function w.r.t. the weight vector, a float numpy array of shape (p,). 
        Hint: you could solve this problem using 1 lines of code
    '''

    dL_dw = dL_da * da_dz * dz_dw

    return dL_dw


#--------------------------
def compute_dL_db(dL_da, da_dz, dz_db):
    '''
        Input:
            dL_da: the local gradient of the loss function w.r.t. the activation, a float scalar value.
            da_dz: the local gradient of the activation a w.r.t. the logit z, a float scalar value. It represents ( d_a / d_z )
            dz_db: the partial gradient of logit z with respect to the bias, a float scalar. It represents (d_z / d_b).
        Output:
            dL_db: the gradient of the loss function w.r.t. the bias, a float scalar. 
        Hint: you could solve this problem using 1 lines of code 
    '''

    dL_db = dL_da * da_dz * dz_db

    return dL_db 


#-----------------------------------------------------------------
# gradient descent 
#-----------------------------------------------------------------

#--------------------------
def update_w(w, dL_dw, alpha=0.001):
    '''
        Input:
            w: the current value of the weight vector, a float numpy array of shape (p,).
            dL_dw: the gradient of the loss function w.r.t. the weight vector, a float numpy array of shape (p,). 
            alpha: the step-size parameter of gradient descent, a float scalar.
        Output:
            w: the updated weight vector, a float numpy array of shape (p,).
        Hint: you could solve this problem using 1 line of code
    '''


    w = w - (alpha * dL_dw)


    return w

#--------------------------
def update_b(b, dL_db, alpha=0.001):
    '''
        Input:
            b: the current value of bias, a float scalar. 
            dL_db: the gradient of the loss function w.r.t. the bias, a float scalar. 
            alpha: the step-size parameter of gradient descent, a float scalar.
        Output:
            b: the updated of bias, a float scalar. 
    '''
    

    b = b - (alpha * dL_db)

    return  b 


#--------------------------
def train(X, Y, alpha=0.001, n_epoch=100):
    '''
        Input:
            X: the feature matrix of training instances, a float numpy array of shape (n,p). Here n is the number of data instance in the training set, p is the number of features/dimensions.
            Y: the labels of training instance, a numpy integer array of shape (n,). The values can be 0 or 1.
            alpha: the step-size parameter of gradient descent, a float scalar.
            n_epoch: the number of passes to go through the training set, an integer scalar.
        Output:
            w: the weight vector trained on the training set, a float numpy array of shape (p,).
            b: the bias, a float scalar. 
    '''

    # initialize weights and biases as 0
    w = np.zeros(X.shape[1])
    b = 0

    for _ in range(n_epoch):
        for x,y in zip(X,Y):
            x = x.T # convert to column vector

            z, a, L = forward(x, y, w, b)

            dL_da, da_dz, dz_dw, dz_db = backward(x, y, a)

            dL_dw = compute_dL_dw(dL_da, da_dz, dz_dw)
            dL_db = compute_dL_db(dL_da, da_dz, dz_db)

            w = update_w(w, dL_dw, alpha)
            b = update_b(b, dL_db, alpha)

    return w, b



#--------------------------
def predict(Xtest, w, b):
    '''
        Input:
            Xtest: the feature matrix of testing instances, a float numpy array of shape (ntest, p). Here n_test is the number of data instance in the test set, p is the number of features/dimensions.
            w: the weight vector of the logistic model, a float numpy array of shape (p,).
            b: the bias value of the logistic model, a float scalar.
        Output:
            Y: the predicted labels of test data, an integer numpy array of length ntest. 
                    If the predicted label is positive, the value is 1. If the label is negative, the value is 0.
            P: the predicted probability of test data to have positive labels, a float numpy array of shape (ntest,). 
                    Each value is between 0 and 1, indicating the probability of the instance having the positive label. 
            Note: If the activation is 0.5, we consider the prediction as positive (instead of negative).
    '''
    n = Xtest.shape[0]
    Y = np.zeros(n) # initialize as all zeros
    P = np.zeros((n, 1))  # Initialize P as a numpy array of zeros

    for i, x in enumerate(Xtest):
        x = x.T # convert to column vector

        z = compute_z(x, w, b)
        a = compute_a(z)
        
        P[i] = a

        if a >= 0.5:
            Y[i] = 1
        else:
            Y[i] = 0

    return Y, P


#-----------------------------------------------------------------
# gradient checking 
#-----------------------------------------------------------------


#--------------------------
def check_dL_da(a, y, delta=1e-10):
    '''
        Input:
            a: the activation value, a float scalar
            y: the label of a training instance, an integer scalar value. The values can be 0 or 1.
            delta: a small number for gradient check, a float scalar.
        Output:
            dL_da: the approximated local gradient of the loss function w.r.t. the activation, a float scalar value.
    '''
    dL_da = (compute_L(a+delta,y) - compute_L(a,y)) / delta
    return dL_da 


#--------------------------
def check_da_dz(z, delta= 1e-7):
    '''
        Input:
            z: the logit value of logistic regression, a float scalar.
            delta: a small number for gradient check, a float scalar.
        Output:
            da_dz: the approximated local gradient of activation a w.r.t. the logit z, a float scalar value.
    '''
    da_dz = (compute_a(z+delta) - compute_a(z)) / delta
    return da_dz 



#--------------------------
def check_dz_dw(x,w, b, delta=1e-7):
    '''
        Input:
            x: the feature vector of a data instance, a float numpy vector of length p. Here p is the number of features/dimensions.
            w: the weight vector of the logistic model, a float numpy vector of length p. 
            b: the bias value of the logistic model, a float scalar.
            delta: a small number for gradient check, a float scalar.
        Output:
            dz_dw: the approximated partial gradient of logit z w.r.t. the weight vector w computed using gradient check, a numpy float vector of length p. 
    '''
    p = x.shape[0] 
    dz_dw = np.zeros(p).reshape(-1, 1)
    for i in range(p):
        d = np.zeros(p).reshape(-1, 1)
        d[i] = delta
        dz_dw[i] = (compute_z(x,w+d, b) - compute_z(x, w, b)) / delta
    return dz_dw


#--------------------------
def check_dz_db(x,w, b, delta=1e-7):
    '''
        Input:
            x: the feature vector of a data instance, a float numpy vector of length p. Here p is the number of features/dimensions.
            w: the weight vector of the logistic model, a float numpy vector of length p. 
            b: the bias value of the logistic model, a float scalar.
            delta: a small number for gradient check, a float scalar.
        Output:
            dz_dw: the approximated partial gradient of logit z w.r.t. the weight vector w computed using gradient check, a numpy float vector of length p. 
            dz_db: the approximated partial gradient of logit z w.r.t. the bias b using gradient check, a float scalar.
    '''
    dz_db = (compute_z(x, w, b+delta) - compute_z(x, w, b)) / delta
    return  dz_db

#--------------------------
def check_dL_dw(x,y,w,b, delta=1e-7):
    '''
        Input:
            x: the feature vector of a training instance, a float numpy vector of length p. Here p is the number of features/dimensions.
            y: the label of a training instance, an integer scalar value. The values can be 0 or 1.
            w: the weight vector, a float numpy vector of length p.
            b: the bias value, a float scalar.
            delta: a small number for gradient check, a float scalar.
        Output:
            dL_dw: the approximated gradient of the loss function w.r.t. the weight vector, a numpy float vector of length p. 
    '''
    p = x.shape[0] # number of features
    dL_dw = np.zeros(p).reshape(-1, 1)
    for i in range(p):
        d = np.zeros(p).reshape(-1, 1)
        d[i] = delta
        dL_dw[i] = (forward(x,y,w+d,b)[-1] - forward(x,y,w,b)[-1]) / delta
    return dL_dw

#--------------------------
def check_dL_db(x,y,w,b, delta=1e-7):
    '''
        Input:
            x: the feature vector of a training instance, a float numpy vector of length p. Here p is the number of features/dimensions.
            y: the label of a training instance, an integer scalar value. The values can be 0 or 1.
            w: the weight vector, a float numpy vector of length p.
            b: the bias value, a float scalar.
            delta: a small number for gradient check, a float scalar.
        Output:
            dL_db: the approximated gradient of the loss function w.r.t. the bias, a float scalar. 
    '''
    dL_db = (forward(x,y,w,b+delta)[-1] - forward(x,y,w,b)[-1]) / delta
    return dL_db