import theano
import numpy as np

def get_prelu_alpha(shape):
# Not sure if this initialisation is exactly what was recommended in the paper
    return theano.shared(np.random.normal(loc=0.0, scale=np.sqrt(2. / \
        np.sum(shape)), size=shape).astype(theano.config.floatX), name='alpha')
    
def prelu(X, alpha):
    pos = 0.5 * (1 + alpha)
    neg = 0.5 * (1 - alpha)
    return pos * X + neg * abs(X)