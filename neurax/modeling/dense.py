import ipypb
import numpy as np
from typing import Union, Optional, Callable, Sequence

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.scipy as jsp
import flax
import flax.linen as nn
import optax
from jax import jit, value_and_grad, random
from jax.tree_util import tree_reduce, tree_map_with_path
from jax.scipy.stats.norm import pdf
from flax.training.train_state import TrainState

from . import Model


class Dense(Model):
    @nn.compact
    def __call__(self, x):
        return nn.Dense(self.channels)(x)


class SmoothDense(Dense):
    smoothing: int = 7

    def train(self, *args, **kwargs):
        smooth_fn = self._make_smooth_fn()

        if 'post_op' in kwargs:
            func = kwargs['post_op']
            def post_op(state):
                state = func(state)
                return state.replace(params=smooth_fn(state.params))
        else:
            def post_op(state):
                return state.replace(params=smooth_fn(state.params))
        kwargs['post_op'] = post_op

        return super().train(*args, **kwargs)

    def _make_smooth_fn(self):
        smooth_r = self.smoothing // 2
        smooth_x = jnp.linspace(-smooth_r, smooth_r, self.smoothing)
        smooth_w = pdf(smooth_x)[..., None]

        @jit
        def smooth_fn(params):
            def smooth_kernel(path, x):
                if path[-1].key == 'kernel':
                    return jsp.signal.convolve(x, smooth_w, mode='same')
                else:
                    return x

            return tree_map_with_path(smooth_kernel, params)

        return smooth_fn

