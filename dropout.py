import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams


def apply_dropout(input, shape, p, rng, rescale=True):
    
    shape = tuple(shape)    
    
    _srng = MRG_RandomStreams(rng.randint(1, 2147462579))    
    
    one = T.constant(1)

    retain_prob = one - p
    
    if rescale:
        input /= retain_prob

    return input * _srng.binomial(shape, p=retain_prob, dtype=input.dtype)