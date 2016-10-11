import theano
from collections import OrderedDict

'''
This is inspired in large part by the Lasagne implementation of Nesterov
Momentum. sgd() takes the symbolic cost function and returns the updates,
apply_nesterov_momentum() changes these updates so that they include Nesterov
Momentum.
'''
def sgd(cost, params, learning_rate):
    # Calc gradients
    grads = theano.grad(cost, params)
    # Store update rules in an OrderedDict as required by Theano
    updates = OrderedDict()
    # Set the updates for general SGD
    for param, grad in zip(params, grads):
        updates[param] = param - learning_rate * grad
    return updates

def apply_nesterov_momentum(updates, momentum, params=None):
    if momentum > 0.:
        # Create updates OrderedDict
        if params is None:
            params = updates.keys()
        updates = OrderedDict(updates)
        # Calculate updates for each param
        for param in params:
            value = param.get_value(borrow=True)
            velocity = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                     broadcastable=param.broadcastable,
                                     name='velocity')
            x = momentum * velocity + updates[param] - param
            updates[velocity] = x
            updates[param] = momentum * x + updates[param]

    return updates