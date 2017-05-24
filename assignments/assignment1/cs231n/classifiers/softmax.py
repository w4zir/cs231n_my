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

  # compute D, C, N
  num_train = X.shape[0]
  num_features = X.shape[1]
  num_classes = W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in range(num_train):
      scores = np.dot(X[i,:],W)
      c_score = -np.max(scores)
      correct_class_score = scores[y[i]]
      all_class_norm_prob = np.exp(scores + c_score)/sum(np.exp(scores + c_score))
      loss -= np.log(all_class_norm_prob[y[i]])

      dldw = all_class_norm_prob.reshape(1 ,-1)
      dldw[0,y[i]] = dldw[0,y[i]] - 1
      dldw = np.dot(X[i].reshape(-1,1),dldw)
      dW += dldw

  loss /= num_train # take average
  loss += reg * np.sum(W*W)
  dW += 2*reg*np.sum(W)
  dW /= num_train
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  # compute D, C, N
  num_train = X.shape[0]
  num_features = X.shape[1]
  num_classes = W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = np.dot(X,W)
  print(num_train)
  c_score = -np.max(scores,axis=1)
  correct_class_score = scores[np.arange(num_train), y]
  # correct_class_norm_prob = np.exp(correct_class_score + c_score)/np.sum(np.exp(scores + c_score.reshape(-1,1)),axis=1)
  all_class_norm_prob = np.exp(scores + c_score.reshape(-1,1))/np.sum(np.exp(scores + c_score.reshape(-1,1)),axis=1)[:,None]
  correct_class_norm_prob = all_class_norm_prob[np.arange(num_train), y]
  print(all_class_norm_prob.shape)
  loss = -np.sum(np.log(correct_class_norm_prob))
#
  dldw = all_class_norm_prob
  dldw[np.arange(num_train), y] = dldw[np.arange(num_train), y] - 1
  dW = np.dot(X.T,dldw)

  loss /= num_train # take average
  loss += reg * np.sum(W*W)
  dW /= num_train
  dW += 2*reg*np.sum(W)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
