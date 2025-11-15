'''
Author: Chih-Kang Huang && chih-kang.huang@hotmail.com
Date: 2025-11-13 08:18:50
LastEditors: Chih-Kang Huang && chih-kang.huang@hotmail.com
LastEditTime: 2025-11-13 08:19:35
FilePath: /steerable/models.py
Description: 


'''
import jax
import jax.numpy as jnp
import equinox as eqx

class FourierControl(eqx.Module):
    """Fourier-based control ansatz: u(t) = a0 + Σ [a_m cos + b_m sin]."""
    a0: jnp.ndarray
    a: jnp.ndarray
    b: jnp.ndarray
    T: float
    A_max: float

    def __init__(self, key, M=6, T=1.0, A_max=1.0, scale=1e-2):
        k1, k2, k3 = jax.random.split(key, 3)
        self.a0 = jax.random.normal(k1, ()) * scale
        self.a  = jax.random.normal(k2, (M,)) * scale / jnp.arange(1, M+1)
        self.b  = jax.random.normal(k3, (M,)) * scale / jnp.arange(1, M+1)
        self.T = T
        self.A_max = A_max

    def __call__(self, t):
        """Evaluate control amplitude at time t ∈ [0, T]."""
        t = jnp.atleast_1d(t)
        freqs = jnp.arange(1, self.a.size + 1)
        cos_terms = jnp.sum(self.a * jnp.cos(2*jnp.pi*freqs[None,:]*t[:,None]/self.T), axis=-1)
        sin_terms = jnp.sum(self.b * jnp.sin(2*jnp.pi*freqs[None,:]*t[:,None]/self.T), axis=-1)
        u = self.a0 + cos_terms + sin_terms
        ## optional amplitude bound
        #u = self.A_max * jnp.tanh(u / self.A_max)
        return u if u.size > 1 else u[0]


class PiecewiseConstantControl(eqx.Module):
    amplitudes: jnp.ndarray  # shape (n_segments,)
    t_final: float
    n_segments: int

    def __call__(self, t: float):
        """Return the control amplitude u(t) for given time t."""
        idx = jnp.clip(
            (t / self.t_final * self.n_segments).astype(int),
            0,
            self.n_segments - 1,
        )
        return self.amplitudes[idx]

    def values(self, times: jnp.ndarray):
        """Convenience method: return u(t) for an array of times."""
        return jax.vmap(self.__call__)(times)