import theano
import numpy as np

def get_prelu_alpha(shape):
    return theano.shared(0.25 * np.ones(shape).astype(theano.config.floatX), name='alpha')
    
def prelu(X, alpha):
    pos = 0.5 * (1 + alpha)
    neg = 0.5 * (1 - alpha)
    return pos * X + neg * abs(X)
