import jax
import jax.numpy as jnp
from jax import jit
from jax.tree_util import tree_reduce


@jit
def mae(preds, targets):
    return jnp.abs(preds - targets).mean()


@jit
def mse(preds, targets):
    return jnp.power(preds - targets, 2).mean()


@jit
def l1_penalty(params):
    return tree_reduce(lambda s, p: s + jnp.abs(p).sum(), params, 0.)


@jit
def l2_penalty(params):
    return tree_reduce(lambda s, p: s + jnp.power(p, 2).sum(), params, 0.)

