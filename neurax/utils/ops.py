from functools import partial

import jax
import jax.numpy as jnp
from jax import jit


@partial(jit, static_argnames=['axis'])
def corr(a, b, axis=None):
    """
    Compute Pearson's correlation along specified axis.

    Parameters
    ----------
    a : np.ndarray
        First input tensor
    b : np.ndarray
        Second input tensor with same shape as `a`
    axis : optional int, default=None
        Axis to take correlation on. If a number, collapses dimension of that
        axis. If None, computes correlation of flattened `a` and `b`, returns
        a single scalar.

    Returns
    -------
    r : np.ndarray or scalar
        The Pearson's r-value for the performed correlations. If axis is None,
        returns a single scalar. If axis is an integer, returns a tensor with
        one dimension less than `a` and `b`.
    """

    a_mean = jnp.mean(a, axis=axis, keepdims=True)
    b_mean = jnp.mean(b, axis=axis, keepdims=True)
    a, b = (a - a_mean), (b - b_mean)
    
    a_sum2 = jnp.power(a, 2).sum(axis=axis, keepdims=True)
    b_sum2 = jnp.power(b, 2).sum(axis=axis, keepdims=True)
    
    a, b = (a / jnp.sqrt(a_sum2)), (b / jnp.sqrt(b_sum2))
    
    return (a * b).sum(axis=axis)

