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


class Model(nn.Module):
    channels: int

    @nn.compact
    def __call__(self, x):
        raise RuntimeError('This is a base class and not meant to be used directly.')

    def train(
        self,
        x: np.ndarray,
        y: np.ndarray,
        loss: Callable,
        penalty: Optional[Callable]=None,
        *,
        alpha: float=1.,
        n_epochs: int=1000,
        lr: float=1e-3,
        post_op: Optional[Callable]=None,
        progress: bool=True,
        verbose: bool=True,
    ):
        """
        Trains a model that predicts `y` from `x`.

        Parameters
        ----------
        x : ndarray
            Input features that computation is performed on
        y : ndarray
            Output tensor that the model should predict from `x`
        loss : callable
            The loss function, called as `loss(preds, targets)` and returning a scalar
        penalty : optional callable
            Extra loss function that will be called on the model parameters rather
            than predictions, returning a scalar
        alpha : optional float
            Relative scaling of penalty loss compared to the main loss
        n_epochs : int
            Number of epochs to train the model
        lr : float
            The learning rate of the Adam optimizer
        post_op : optional callable
            A function for transforming model state that is applied once after
            every gradient update
        progress : bool
            Whether to show a progress bar of how many epochs trained
        verbose : bool
            Whether to display loss value during training

        Returns
        -------
        params : pytree
            A frozen dictionary containing the parameters of the trained model
        """

        key_init = random.PRNGKey(0)
        assert self.channels == y.shape[-1]

        tx = optax.adam(lr)
        params = self.init(key_init, x)
        state = TrainState.create(apply_fn=self.apply, params=params, tx=tx)
        grad_fn = jit(value_and_grad(self._make_loss_fn(loss, penalty, alpha)))

        epochs = ipypb.irange(n_epochs) if progress else range(n_epochs)
        for epoch in epochs:
            loss, grads = grad_fn(state.params, x, y)
            state = state.apply_gradients(grads=grads)
            if post_op:
                state = post_op(state)

            if verbose and (epoch+1) % 100 == 0:
                print(f'Epoch {epoch+1:04d} --> loss: {loss}')

        return state.params
    
    def eval(self, params, x, y):
        z = self.apply(params, x)
        r = corr(z, y, axis=1).mean(0)
        return r

    def _make_loss_fn(self, loss, penalty, alpha=1.):
        @jit
        def loss_fn(params, x, y):
            pred = self.apply(params, x)
            if penalty and alpha != 0.:
                return loss(pred, y) + alpha * penalty(params)
            else:
                return loss(pred, y)

        return loss_fn

