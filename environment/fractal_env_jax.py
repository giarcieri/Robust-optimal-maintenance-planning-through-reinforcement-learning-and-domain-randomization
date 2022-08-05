import jax
import chex
from typing import Tuple, Union, Optional, Dict
from functools import partial
from numpyro.distributions import StudentT
from .hmm_AR_k_Tstud import HMMStates, TruncatedNormalEmissionsAR_k

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
        self.dtype = jnp.int32

    def sample(self, rng: chex.PRNGKey) -> chex.Array:
        """Sample random action uniformly from set of categorical choices."""
        return jax.random.randint(
            rng, shape=self.shape, minval=0, maxval=self.n
        ).astype(self.dtype)

    def contains(self, x: jnp.int32) -> bool:
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

    @partial(jax.jit, static_argnums=(0,)) 
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

    @partial(jax.jit, static_argnums=(0,))
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
        def true_fun(key):
            return deterioration_process(key, state, obs, params)
        def false_fun(key):
            return repair_process(key, state, obs, action, params)
        obs = jax.lax.cond(action==0, true_fun, false_fun, key)
        return obs, state, reward, False, {}

    def reset_env(
        self, key: chex.PRNGKey, params: Dict[str, jnp.ndarray]
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Environment-specific reset."""
        # sample initial state
        init_probs = params['init_probs']
        state = jax.random.choice(key=key, a=4, p=jnp.asarray(init_probs))
        # sample initial obs
        obs = init_process(key, state, params)
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
        return Box(-1e3, 0, shape=(1,), dtype=jnp.float32)

    def state_space(self) -> Discrete:
        """State space of the environment."""
        return Discrete(4)

def _deterioration_process(key, state, obs, params):
    mu_d, sigma_d, nu_d = params['mu_d'][state], params['sigma_d'][state], params['nu_d'][state]
    return StudentT(df=nu_d, loc=mu_d, scale=sigma_d).sample(key)

def deterioration_process(key, state, obs, params):
    sample = _deterioration_process(key, state, obs, params)
    def true_fun(key):
        return sample + obs
    def false_fun(key):
        key, _ = jax.random.split(key)
        return _deterioration_process(key, state, obs, params)
    pred = lambda x: x < -obs
    return jax.lax.cond(pred(sample), true_fun, false_fun, key)

def _repair_process(key, state, obs, action, params): 
    mu_r, sigma_r, nu_r, k = params['mu_r'][state], params['sigma_r'][state], params['nu_r'][state], params['k'][action-1]
    return StudentT(df=nu_r, loc=k*obs + mu_r, scale=sigma_r).sample(key)

def repair_process(key, state, obs, action, params): 
    sample = _repair_process(key, state, obs, action, params)
    def true_fun(key):
        return sample
    def false_fun(key):
        key, _ = jax.random.split(key)
        return _repair_process(key, state, obs, action, params)
    pred = lambda x: x < 0.0
    return jax.lax.cond(pred(sample), true_fun, false_fun, key)
    

def _init_process(key, state, params):
    mu_init, sigma_init, nu_init = params['mu_init'][state], params['sigma_init'][state], params['nu_init'][state]
    return StudentT(df=nu_init, loc=mu_init, scale=sigma_init).sample(key)

def init_process(key, state, params):
    sample = _init_process(key, state, params)
    def true_fun(key):
        return sample
    def false_fun(key):
        key, _ = jax.random.split(key)
        return _init_process(key, state, params)
    pred = lambda x: x < 0.0
    return jax.lax.cond(pred(sample), true_fun, false_fun, key)
    

def sample_params(key, trace): 
    n_samples = trace['p_transition'].shape[0]
    index_sample = jax.random.choice(key=key, a=n_samples)

    transition_matrices = trace['p_transition'][index_sample]
    init_probs = trace['init_probs'][index_sample]

    mu_d = trace['mu_d'][index_sample]
    sigma_d = trace['sigma_d'][index_sample]
    nu_d = trace['nu_d'][index_sample]

    mu_r = trace['mu_r'][index_sample]
    sigma_r = trace['sigma_r'][index_sample]
    nu_r = trace['nu_r'][index_sample]

    mu_init = trace['mu_init'][index_sample]
    sigma_init = trace['sigma_init'][index_sample]
    nu_init = trace['nu_init'][index_sample]

    k = trace['k'][index_sample]
    params = {
        'init_probs': init_probs,
        'p_transition': transition_matrices, 
        'mu_d': mu_d,
        'sigma_d': sigma_d, 
        'nu_d': nu_d,
        'mu_r': mu_r,
        'sigma_r': sigma_r,
        'nu_r': nu_r,
        'mu_init': mu_init,
        'sigma_init': sigma_init,
        'nu_init': nu_init,
        'k': k
    }
    return jax.tree_map(jnp.asarray, params)

def sample_mean_params(trace): 
    transition_matrices = trace['p_transition'].mean(0)
    init_probs = trace['init_probs'].mean(0)

    mu_d = trace['mu_d'].mean(0)
    sigma_d = trace['sigma_d'].mean(0)
    nu_d = trace['nu_d'].mean(0)

    mu_r = trace['mu_r'].mean(0)
    sigma_r = trace['sigma_r'].mean(0)
    nu_r = trace['nu_r'].mean(0)

    mu_init = trace['mu_init'].mean(0)
    sigma_init = trace['sigma_init'].mean(0)
    nu_init = trace['nu_init'].mean(0)

    k = trace['k'].mean(0)
    params = {
        'init_probs': init_probs,
        'p_transition': transition_matrices, 
        'mu_d': mu_d,
        'sigma_d': sigma_d, 
        'nu_d': nu_d,
        'mu_r': mu_r,
        'sigma_r': sigma_r,
        'nu_r': nu_r,
        'mu_init': mu_init,
        'sigma_init': sigma_init,
        'nu_init': nu_init,
        'k': k
    }
    return jax.tree_map(jnp.asarray, params)

