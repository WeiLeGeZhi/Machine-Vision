from builtins import range
import numpy as np
from random import shuffle
# from past.builtins import xrange

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
    # TODO: 使用显式循环计算softmax损失及其梯度。
    # 将损失和梯度分别保存在loss和dW中。
    # 如果你不小心，很容易遇到数值不稳定的情况。 
    # 不要忘了正则化！                                                           
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    scores=X@W
    num_train=X.shape[0]
    num_labels=W.shape[1]

    for i in range(num_train):
        scr_sum=np.exp(scores[i]-np.max(scores[i]))/np.sum(np.exp(scores[i]-np.max(scores[i])))
        loss-=np.log(scr_sum[y[i]])
        for j in range(num_labels):
            dW[:,j]+=X[i] * scr_sum[j]
        dW[:,y[i]]-=X[i]
    
    loss/=num_train
    dW/=num_train
    loss+=reg*np.sum(W*W)
    dW+=2*reg*W
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

    #############################################################################
    # TODO: 不使用显式循环计算softmax损失及其梯度。
    # 将损失和梯度分别保存在loss和dW中。
    # 如果你不小心，很容易遇到数值不稳定的情况。 
    # 不要忘了正则化！
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train=X.shape[0]
    scores=X@W
    scores-=np.max(scores, axis=1, keepdims=True)
    scr_sum=np.sum(np.exp(scores), axis=1, keepdims=True)
    p=np.exp(scores)/scr_sum

    loss=np.sum(-np.log(p[np.arange(num_train),y]))

    ind=np.zeros_like(p)
    ind[np.arange(num_train),y]=1
    dW=X.T@(p-ind)

    loss/=num_train
    loss+=0.5*reg*np.sum(W*W)
    dW/=num_train
    dW+=reg*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
