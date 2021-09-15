from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    for i in range(X.shape[0]):
      score = []
      Z = X[i].dot(W)
      Z -= np.max(Z)
      for j in range(W.shape[1]):
        score.append(np.exp(Z[j])/np.sum(np.exp(Z)))
      loss += -np.log(score[y[i]])
      for j in range(W.shape[1]):
        if j==y[i]:
          dW[:,j] -= X[i]*(1-score[j])
        else:
          dW[:,j] += X[i]*score[j]
    loss /= X.shape[0]
    dW /= X.shape[0]
    loss += reg*np.sum(W*W)
    dW += 2*reg*W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0]
    num_class = W.shape[1]
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    Z = X.dot(W)
    Z -= np.max(Z)
    score = np.zeros((num_train, num_class))
    score = np.exp(Z)/np.matrix(np.sum(np.exp(Z), axis=1)).T
    loss = -np.sum(np.log(score[np.arange(num_train), y]))
    score[np.arange(num_train), y] -=1
    dW = X.T.dot(score)
    loss /= num_train
    loss += reg*np.sum(W*W)
    dW /= num_train
    dW += 2*reg*W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
