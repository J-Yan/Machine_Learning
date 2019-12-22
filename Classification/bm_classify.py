import numpy as np


def binary_train(X, y, loss="perceptron", w0=None, b0=None, step_size=0.5, max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the
    number of training points and D is the dimensionality of features
    - y: binary training labels, a N dimensional numpy array where
    N is the number of training points, indicating the labels of
    training data
    - loss: loss type, either perceptron or logistic
    - step_size: step size (learning rate)
	- max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: D-dimensional vector, a numpy array which is the weight
    vector of logistic or perceptron regression
    - b: scalar, which is the bias of logistic or perceptron regression
    """
    N, D = X.shape
    assert len(np.unique(y)) == 2


    w = np.zeros(D)
    if w0 is not None:
        w = w0

    b = 0
    if b0 is not None:
        b = b0

    if loss == "perceptron":
        ############################################
        # TODO 1 : Edit this if part               #
        #          Compute w and b here            #
        y_minus1and1 = 2*(y-0.5)
        X_1 = np.concatenate((X,np.array([np.ones(N)]).T),axis=1)
        w_b = np.append(np.transpose(w),b)
        for i in range(max_iterations):
            # ywx = np.multiply(np.sum(np.multiply(w_b,X_1),axis=1),y_minus1and1)
            ywx = np.multiply(np.dot(X_1,w_b),y_minus1and1)
            ywx_0 = np.where(ywx>0, 0, y_minus1and1)
            s = np.dot(ywx_0,X_1)
            w_b += step_size*(1/N)*s

        w = np.transpose(w_b[:-1])
        b = w_b[-1]
        ############################################


    elif loss == "logistic":
        ############################################
        # TODO 2 : Edit this if part               #
        #          Compute w and b here            #
        y_minus1and1 = 2*(y-0.5)
        X_1 = np.concatenate((X,np.array([np.ones(N)]).T),axis=1)
        w_b = np.append(np.transpose(w),b)
        for i in range(max_iterations):
            # sigyx = np.multiply(X_1.T,np.multiply(y_minus1and1,sigmoid(-np.multiply(np.sum(np.multiply(w_b,X_1),axis=1),y_minus1and1)))).T
            sigyx = np.dot(X_1.T,np.multiply(y_minus1and1,sigmoid(-np.multiply(np.dot(X_1,w_b),y_minus1and1)))).T
            w_b += step_size/N*sigyx

        w = np.transpose(w_b[:-1])
        b = w_b[-1]
        ############################################


    else:
        raise "Loss Function is undefined."

    assert w.shape == (D,)
    return w, b

def sigmoid(z):

    """
    Inputs:
    - z: a numpy array or a float number

    Returns:
    - value: a numpy array or a float number after computing sigmoid function value = 1/(1+exp(-z)).
    """

    ############################################
    # TODO 3 : Edit this part to               #
    #          Compute value                   #
    value = 1/(1+np.exp(-z))
    ############################################

    return value

def binary_predict(X, w, b, loss="perceptron"):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the
    number of training points and D is the dimensionality of features
    - w: D-dimensional vector, a numpy array which is the weight
    vector of your learned model
    - b: scalar, which is the bias of your model
    - loss: loss type, either perceptron or logistic

    Returns:
    - preds: N dimensional vector of binary predictions: {0, 1}
    """
    N, D = X.shape

    if loss == "perceptron":
        ############################################
        # TODO 4 : Edit this if part               #
        #          Compute preds                   #
        X_1 = np.concatenate((X,np.array([np.ones(N)]).T),axis=1)
        w_b = np.append(np.transpose(w),b)
        wx = np.sum(np.multiply(X_1,w_b),axis=1)
        preds = np.where(wx>0, 1.0, 0.0)
        ############################################


    elif loss == "logistic":
        ############################################
        # TODO 5 : Edit this if part               #
        #          Compute preds                   #
        X_1 = np.concatenate((X,np.array([np.ones(N)]).T),axis=1)
        w_b = np.append(np.transpose(w),b)
        wx = np.sum(np.multiply(X_1,w_b),axis=1)
        preds = np.where(sigmoid(wx)>=0.5, 1.0, 0.0)
        ############################################


    else:
        raise "Loss Function is undefined."


    assert preds.shape == (N,)
    return preds



def multiclass_train(X, y, C,
                     w0=None,
                     b0=None,
                     gd_type="sgd",
                     step_size=0.5,
                     max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the
    number of training points and D is the dimensionality of features
    - y: multiclass training labels, a N dimensional numpy array where
    N is the number of training points, indicating the labels of
    training data
    - C: number of classes in the data
    - gd_type: gradient descent type, either GD or SGD
    - step_size: step size (learning rate)
    - max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: C-by-D weight matrix of multinomial logistic regression, where
    C is the number of classes and D is the dimensionality of features.
    - b: bias vector of length C, where C is the number of classes
    """

    N, D = X.shape

    w = np.zeros((C, D))
    if w0 is not None:
        w = w0

    b = np.zeros(C)
    if b0 is not None:
        b = b0

    # one-hot matrix
    y_OH = np.zeros((N,C))
    y_OH[np.arange(N),y] = 1

    np.random.seed(42)
    if gd_type == "sgd":
        ############################################
        # TODO 6 : Edit this if part               #
        #          Compute w and b                 #
        X_1 = np.concatenate((X,np.array([np.ones(N)]).T),axis=1)
        w_b = np.concatenate((w,np.array([b]).T),axis=1)

        for i in range(max_iterations):
            r = np.random.choice(N)
            wx = np.dot(w_b,X_1[r].reshape(D+1,1))
            wx -= wx.max()
            wx_exp = np.exp(wx)
            P = wx_exp/np.sum(wx_exp)
            P[y[r]] -= 1
            Px = np.dot(P.reshape((C,1)),X_1[r].reshape(1,D+1))
            w_b -= step_size*Px
        w = w_b[:,:-1]
        b = w_b[:,-1]
        ############################################


    elif gd_type == "gd":
        ############################################
        # TODO 7 : Edit this if part               #
        #          Compute w and b                 #
        X_1 = np.concatenate((X,np.array([np.ones(N)]).T),axis=1)
        w_b = np.concatenate((w,np.array([b]).T),axis=1)
        for i in range(max_iterations):

            wx = np.dot(w_b,X_1.T)
            wx -= wx.max()
            wx_exp = np.exp(wx)
            P = wx_exp/np.sum(wx_exp,axis=0) - y_OH.T
            Px = np.dot(P,X_1)
            w_b -= step_size*Px/N
        w = w_b[:,:-1]
        b = w_b[:,-1]
        ############################################


    else:
        raise "Type of Gradient Descent is undefined."


    assert w.shape == (C, D)
    assert b.shape == (C,)

    return w, b


def multiclass_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the
    number of training points and D is the dimensionality of features
    - w: weights of the trained multinomial classifier, C-by-D
    - b: bias terms of the trained multinomial classifier, length of C

    Returns:
    - preds: N dimensional vector of multiclass predictions.
    Outputted predictions should be from {0, C - 1}, where
    C is the number of classes
    """
    N, D = X.shape
    ############################################
    # TODO 8 : Edit this part to               #
    #          Compute preds                   #
    preds = np.zeros(N)
    for i in range(N):
        wx = np.sum(np.multiply(w,X[i]),axis=1)+b
        preds[i] = np.argmax(wx)
    ############################################

    assert preds.shape == (N,)
    return preds
