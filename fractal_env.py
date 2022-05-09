import gym
import jax
import haiku as hk
import pymc3 as pm

from jax import numpy as jnp


class FractalEnv(gym.Env):

    def __init__(self, trace, seed, reward_matrix, randomization=True) -> None:
        """Environment that simulates the POMDP railway maintenance planning problem based on fractal values observations 
        and 3 actions.

        Args:
            trace (PyMC3 Multitrace or dict): trace containing parameter posterior distribution
            seed (int): seed for rng
            reward_matrix (jnp.ndarray): reward matrix with shape (n_actions, n_states)
            randomization (bool, optional): domain randomization. Defaults to True.
        """
        super().__init__()

        self._rng = hk.PRNGSequence(seed)
        self.n_samples = trace['p_transition'].shape[0]
        self.n_states = trace['init_probs'].shape[1]
        self.NegativeStudentT = pm.Bound(pm.StudentT, upper=0.0).dist

        self.trace = trace
        self.randomization = randomization
        self.initialized = False
        self.reward_matrix = reward_matrix

        self.action_space = jnp.array([0, 1, 2])
        self.state_space = jnp.arange(self.n_states)
        self.observation_space = lambda x: x < 0
        
    def sample_params(self): 
        index_sample = jax.random.choice(key=next(self._rng), a=self.n_samples)

        self.transition_matrices = self.trace['p_transition'][index_sample]
        self.init_probs = self.trace['init_probs'][index_sample]

        self.mu_d = self.trace['mu_d'][index_sample]
        self.sigma_d = self.trace['sigma_d'][index_sample]
        self.nu_d = self.trace['nu_d'][index_sample]

        self.mu_r = self.trace['mu_r'][index_sample]
        self.sigma_r = self.trace['sigma_r'][index_sample]
        self.nu_r = self.trace['nu_r'][index_sample]

        self.mu_init = self.trace['mu_init'][index_sample]
        self.sigma_init = self.trace['sigma_init'][index_sample]
        self.nu_init = self.trace['nu_init'][index_sample]

        self.k = self.trace['k'][index_sample]

    def sample_initial_state(self):
        state = jax.random.choice(key=next(self._rng), a=self.n_states, p=jnp.asarray(self.init_probs))
        return state

    def sample_initial_obs(self, state):
        mu_init = self.mu_init[state]
        sigma_init = self.sigma_init[state]
        nu_init = self.nu_init[state]
        obs = self.NegativeStudentT(mu=mu_init, sigma=sigma_init, nu=nu_init).random()
        return jnp.asarray(obs)

    def reset(self):
        if not self.initialized:
            self.sample_params()
            self.initialized = True
        elif self.randomization:
            self.sample_params()
        state = self.sample_initial_state()
        obs = self.sample_initial_obs(state)
        self.state = state
        self.obs = obs
        return obs

    def sample_new_state(self, action):
        transition_probs = jnp.asarray(self.transition_matrices[action, self.state])
        new_state = jax.random.choice(key=next(self._rng), a=self.n_states, p=transition_probs.squeeze()) 
        return new_state

    def deterioration_process(self):
        mu_d, sigma_d, nu_d = self.mu_d[self.state], self.sigma_d[self.state], self.nu_d[self.state]
        DetStudentT = pm.Bound(pm.StudentT, upper=float(-self.obs)).dist
        new_obs = DetStudentT(mu=mu_d, sigma=sigma_d, nu=nu_d).random() + self.obs
        return jnp.asarray(new_obs)

    def repair_process(self, action):
        mu_r, sigma_r, nu_r = self.mu_r[self.state], self.sigma_r[self.state], self.nu_r[self.state]
        k = self.k[action-1]
        new_obs = self.NegativeStudentT(mu=k*self.obs + mu_r, sigma=sigma_r, nu=nu_r).random()
        return jnp.asarray(new_obs)
         

    def step(self, action):
        action_err_msg = f"{action!r} ({type(action)}) invalid"
        assert action in self.action_space, action_err_msg
        assert self.state is not None, "Call reset before using step method."

        reward = self.reward_matrix[action, self.state]

        new_state = self.sample_new_state(action)
        state_err_msg = f"{new_state!r} ({type(new_state)}) invalid"
        assert new_state in self.state_space, state_err_msg
        self.state = new_state

        if action == 0:
            new_obs = self.deterioration_process()
        elif action == 1 or action == 2:
            new_obs = self.repair_process(action)
        obs_err_msg = f"{new_obs!r} ({type(new_obs)}) invalid"
        assert self.observation_space(new_obs), obs_err_msg
        self.obs = new_obs

        done = False
        info = {}
        return new_obs, reward, done, info
