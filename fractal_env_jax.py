import jax
import chex
import pymc3 as pm
from typing import Tuple, Union, Optional, Dict
from functools import partial

from jax import numpy as jnp

class Discrete(object):
    """
    Minimal jittable class for discrete gymnax spaces.
    Taken from gymnax, cannot currently install for dependencies conflicts.
    """

    def __init__(self, num_categories: int):
        assert num_categories >= 0
        self.n = num_categories
        self.shape = ()
        self.dtype = jnp.int_

    def sample(self, rng: chex.PRNGKey) -> chex.Array:
        """Sample random action uniformly from set of categorical choices."""
        return jax.random.randint(
            rng, shape=self.shape, minval=0, maxval=self.n
        ).astype(self.dtype)

    def contains(self, x: jnp.int_) -> bool:
        """Check whether specific object is within space."""
        range_cond = jnp.logical_and(x >= 0, x < self.n)
        return range_cond


class Box(object):
    """
    Minimal jittable class for array-shaped gymnax spaces.
    Taken from gymnax, cannot currently install for dependencies conflicts.
    """

    def __init__(
        self,
        low: float,
        high: float,
        shape: Tuple[int],
        dtype: jnp.dtype = jnp.float32,
    ):
        self.low = low
        self.high = high
        self.shape = shape
        self.dtype = dtype

    def sample(self, rng: chex.PRNGKey) -> chex.Array:
        """Sample random action uniformly from 1D continuous range."""
        return jax.random.uniform(
            rng, shape=self.shape, minval=self.low, maxval=self.high
        ).astype(self.dtype)

    def contains(self, x: jnp.int_) -> bool:
        """Check whether specific object is within space."""
        range_cond = jnp.logical_and(
            jnp.all(x >= self.low), jnp.all(x <= self.high)
        )
        return range_cond


class Environment(object):
    """
    Jittable abstract base class for all gymnax Environments. 
    Taken from gymnax, cannot currently install for dependencies conflicts.
    """

    #@partial(jax.jit, static_argnums=(0,)) 
    def step(
        self,
        key: chex.PRNGKey,
        state: jnp.ndarray,
        obs: jnp.ndarray,
        action: Union[int, float],
        params: Dict[str, jnp.ndarray],
    ) -> Tuple[jnp.ndarray, jnp.ndarray, float, bool, Dict]:
        """Performs step transitions in the environment."""
        key, key_reset = jax.random.split(key)
        obs_st, state_st, reward, done, info = self.step_env(
            key, state, obs, action, params
        )
        obs_re, state_re = self.reset_env(key_reset, params)
        # Auto-reset environment based on termination
        state = jax.tree_multimap(
            lambda x, y: jax.lax.select(done, x, y), state_re, state_st
        )
        obs = jax.lax.select(done, obs_re, obs_st)
        return obs, state, reward, done, info

    #@partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey, params: Dict[str, jnp.ndarray]
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Performs resetting of environment."""
        obs, state = self.reset_env(key, params)
        return obs, state

    def step_env(
        self,
        key: chex.PRNGKey,
        state: jnp.ndarray,
        obs: jnp.ndarray,
        action: Union[int, float],
        params: Dict[str, jnp.ndarray],
    ) -> Tuple[jnp.ndarray, jnp.ndarray, float, bool, Dict]:
        """Environment-specific step transition."""
        raise NotImplementedError

    def reset_env(
        self, key: chex.PRNGKey, params: Dict[str, jnp.ndarray]
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Environment-specific reset."""
        raise NotImplementedError

    def get_obs(self, state: jnp.ndarray) -> jnp.ndarray:
        """Applies observation function to state."""
        raise NotImplementedError

    def is_terminal(self, state: jnp.ndarray, params: Dict[str, jnp.ndarray]) -> bool:
        """Check whether state transition is terminal."""
        raise NotImplementedError

    @property
    def name(self) -> str:
        """Environment name."""
        return type(self).__name__

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        raise NotImplementedError

    def action_space(self, params: Dict[str, jnp.ndarray]):
        """Action space of the environment."""
        raise NotImplementedError

    def observation_space(self, params: Dict[str, jnp.ndarray]):
        """Observation space of the environment."""
        raise NotImplementedError

    def state_space(self, params: Dict[str, jnp.ndarray]):
        """State space of the environment."""
        raise NotImplementedError


class FractalEnv(Environment):
    """
    Jax-compatible implementation of fractal values environment.
    """
    def __init__(self, reward_matrix):
        super().__init__()
        self.reward_matrix = reward_matrix

    def step_env(
        self,
        key: chex.PRNGKey,
        state: jnp.ndarray,
        obs: jnp.ndarray,
        action: Union[int, float],
        params: Dict[str, jnp.ndarray],
    ) -> Tuple[jnp.ndarray, jnp.ndarray, float, bool, Dict]:
        """Environment-specific step transition."""
        # sample reward
        reward = self.reward_matrix[action, state]
        # sample new state
        transition_matrices = params['p_transition']
        transition_probs = jnp.asarray(transition_matrices[action, state])
        state = jax.random.choice(key=key, a=4, p=transition_probs.squeeze()) 
        # sample new obs
        if action == 0:
            obs = deterioration_process(state, obs, params)
        else:
            obs = repair_process(state, obs, action, params)
        return obs, state, reward, False, {}

    def reset_env(
        self, key: chex.PRNGKey, params: Dict[str, jnp.ndarray]
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Environment-specific reset."""
        # sample initial state
        init_probs = params['init_probs']
        state = jax.random.choice(key=key, a=4, p=jnp.asarray(init_probs))
        # sample initial obs
        obs = init_process(state, params)
        return obs, state

    @property
    def name(self) -> str:
        """Environment name."""
        return "Fractal-values"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 3

    def action_space(self) -> Discrete:
        """Action space of the environment."""
        return Discrete(3)

    def observation_space(self) -> Box:
        """Observation space of the environment."""
        return Box(jnp.NINF, 0, shape=(1,), dtype=jnp.float32)

    def state_space(self) -> Discrete:
        """State space of the environment."""
        return Discrete(4)

def deterioration_process(state, obs, params):
    mu_d, sigma_d, nu_d = params['mu_d'][state], params['sigma_d'][state], params['nu_d'][state]
    DetStudentT = pm.Bound(pm.StudentT, upper=float(-obs)).dist
    new_obs = DetStudentT(mu=mu_d, sigma=sigma_d, nu=nu_d).random() + obs
    return jnp.asarray(new_obs)

def repair_process(state, obs, action, params): 
    mu_r, sigma_r, nu_r, k = params['mu_r'][state], params['sigma_r'][state], params['nu_r'][state], params['k'][action-1]
    NegativeStudentT = pm.Bound(pm.StudentT, upper=0.0).dist
    new_obs = NegativeStudentT(mu=k*obs + mu_r, sigma=sigma_r, nu=nu_r).random()
    return jnp.asarray(new_obs)

def init_process(state, params):
    mu_init, sigma_init, nu_init = params['mu_init'][state], params['sigma_init'][state], params['nu_init'][state]
    NegativeStudentT = pm.Bound(pm.StudentT, upper=0.0).dist
    obs = NegativeStudentT(mu=mu_init, sigma=sigma_init, nu=nu_init).random()
    return jnp.asarray(obs)