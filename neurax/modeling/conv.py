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

from neurax.utils.ops import corr
from . import Model


class Conv(Model):
    width: int
    padding: Union[str, tuple[int, int]] = 'CAUSAL'

    @nn.compact
    def __call__(self, x):
        return nn.Conv(
            self.channels, [self.width], padding=self.padding, name='conv'
        )(x)
    

class PointConv(Model):
    lag: int = 0

    @nn.compact
    def __call__(self, x):
        if self.lag > 0:
            x = x[self.lag:]
        elif self.lag < 0:
            x = x[:self.lag]

        return nn.Conv(
            self.channels, [1], padding='VALID', name='conv',
        )(x)


class SmoothConv(Conv):
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
        smooth_w = (pdf(smooth_x) * pdf(smooth_x[:, None]))[..., None]

        @jit
        def smooth_fn(params):
            def smooth_kernel(path, x):
                if path[-1].key == 'kernel':
                    return jsp.signal.convolve(x, smooth_w, mode='same')
                else:
                    return x

            return tree_map_with_path(smooth_kernel, params)

        return smooth_fn


class LowRankConv(Conv):
    rank: int = 4

    @nn.compact
    def __call__(self, x):
        freqs = x.shape[-1]
        x = x.swapaxes(1, 2)

        kernel_init = nn.initializers.lecun_normal()
        bias_init = lambda key: jnp.zeros(())

        result = []
        for ch in range(self.channels):
            kernel_t = self.param(
                f'ch{ch:04d}_kernel_t',
                lambda key: kernel_init(key, (self.width, self.rank))
            )
            kernel_f = self.param(
                f'ch{ch:04d}_kernel_f',
                lambda key: kernel_init(key, (self.rank, freqs))
            )
            bias = self.param(
                f'ch{ch:04d}_bias',
                lambda key: jnp.zeros(())
            )
            
            kernel = jnp.dot(kernel_t, kernel_f).T[None]
            output = lax.conv_general_dilated(x, kernel, [1], self.padding).squeeze(1) + bias
            result.append(output)

        return jnp.stack(result, axis=-1)

