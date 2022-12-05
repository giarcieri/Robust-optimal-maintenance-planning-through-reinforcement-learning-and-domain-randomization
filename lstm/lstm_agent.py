import collections
from typing import Optional
import chex
import distrax
import optax
import functools

from .lstm import *

Params = collections.namedtuple("Params", "pi q1 q2 q1_target q2_target")
OptStates = collections.namedtuple("OptStates", "pi q1 q2")

@jax.jit
def inner1d(X, Y):
  return (X * Y).sum(-1)

class LSTMActor():
    """LSTM Policy Network"""

    def __init__(
        self, 
        rng: chex.PRNGKey,
        dummy_obs: jnp.ndarray,
        act_dim: int = 3, 
        hidden_sizes: Iterable[int] = [100, 100],
    ):

        def nn_func():
            return apply_DeepLSTM(hidden_units=hidden_sizes, output_units=act_dim)
        
        self.nn = hk.multi_transform(nn_func)
        self._init_params = self.nn.init(rng, dummy_obs)

    def __call__(
        self,
        rng: chex.PRNGKey,
        obs: jnp.ndarray,
        params: hk.Params,
        deterministic: bool = False,
    ):
        preds = self.nn.apply.forward(params, rng, obs) #.squeeze()?
        categorical_dist = distrax.Categorical(logits=preds)
        log_probs = jnp.log(categorical_dist.probs)
        if deterministic:
            pi_action = log_probs.argmax(-1)
            log_pi = log_probs.max(-1)
        else:
            pi_action, log_pi = categorical_dist.sample_and_log_prob(seed=rng)
        return pi_action, log_pi, categorical_dist.probs

    @property
    def init_params(
        self,
    ):
        return self._init_params

    

class LSTMCritic():
    """LSTM Critic Network that computes q-values"""

    def __init__(
        self,
        rng: chex.PRNGKey,
        dummy_obs: jnp.ndarray,
        hidden_sizes: Iterable[int] = [100, 100],
        act_dim: Optional[int] = 3
    ):
        def nn_func():
            return apply_DeepLSTM(hidden_units=hidden_sizes, output_units=act_dim)

        self.nn = hk.multi_transform(nn_func)
        self._init_params = self.nn.init(rng, dummy_obs)

    @property
    def init_params(
        self,
    ):
        return self._init_params

    def __call__(
        self,
        rng: chex.PRNGKey,
        obs: jnp.ndarray,
        #act: jnp.ndarray,
        params: hk.Params,
    ):
        #inputs = jnp.concatenate([obs, act], axis=-1)
        q_values = self.nn.apply.forward(params, rng, obs)#[:, act]
        return q_values#.squeeze()



class LSTMActorCritic():
    """Class to combine actor and critc networks based on LSTM"""

    def __init__(
        self,
        rng: chex.PRNGKey,
        dummy_obs_actor: jnp.ndarray,
        dummy_obs_critic: jnp.ndarray,
        obs_dim: int = 1,
        act_dim: int = 3,
        hidden_sizes: Iterable[int] = [100, 100],
    ):
        rng1, rng2, rng3 = jax.random.split(rng, num=3)
        self.pi = LSTMActor(
            rng = rng1,
            dummy_obs = dummy_obs_actor,
            act_dim = act_dim, 
            hidden_sizes = hidden_sizes,
        )
        self.q1 = LSTMCritic(
            rng = rng2,
            dummy_obs = dummy_obs_critic,
            hidden_sizes = hidden_sizes,
            act_dim = act_dim,
        )
        self.q2 = LSTMCritic(
            rng = rng3,
            dummy_obs = dummy_obs_critic,
            hidden_sizes = hidden_sizes,
            act_dim = act_dim,
        )

    def init_params(
        self,
    ):
        return self.pi.init_params, self.q1.init_params, self.q2.init_params
    
    def act(
        self,
        rng: chex.PRNGKey,
        obs: jnp.ndarray,
        params: hk.Params,
        deterministic: bool = False,
    ):
        action, _, _ = self.pi(rng, obs, params, deterministic)
        return action

class LSTMSAC():
    """Soft-Actor-Critic method based on LSTM networks"""

    def __init__(
        self,
        rng: chex.PRNGKey,
        dummy_obs_actor: jnp.ndarray,
        dummy_obs_critic: jnp.ndarray,
        obs_dim: int = 1,
        act_dim: int = 3,
        hidden_sizes: Iterable[int] = [100, 100],
        learning_rate: float = 1e-3,
        polyak: float = 0.995,
    ):

        self.ac = LSTMActorCritic(
            rng = rng,
            dummy_obs_actor = dummy_obs_actor,
            dummy_obs_critic = dummy_obs_critic,
            obs_dim = obs_dim,
            act_dim = act_dim,
            hidden_sizes = hidden_sizes,
        )
        self.optimizer = optax.adam(learning_rate)
        self.polyak = polyak
        self.act_dim = act_dim

    def init_params(
        self,
    ):
        pi_params, q1_params, q2_params = self.ac.init_params()
        return Params(pi_params, q1_params, q2_params, q1_params, q2_params)

    def init_opt_state(
        self,
        params: Params,
    ):
        pi_opt_state = self.optimizer.init(params.pi)
        q1_opt_state = self.optimizer.init(params.q1)
        q2_opt_state = self.optimizer.init(params.q2)
        return OptStates(pi_opt_state, q1_opt_state, q2_opt_state)

    @functools.partial(jax.jit, static_argnums=(0,))
    def bellman_backup(
        self,
        rng: chex.PRNGKey,
        data: Tuple[jnp.ndarray],
        params: Params,
        alpha: float = 0.2,
    ):
        rng1, rng2, rng3 = jax.random.split(rng, num=3)
        _, _, r_t, discount_t, obs_t = data
        # sample next action
        #a_t, logp_a_t = self.ac.pi(rng1, obs_t, params.pi, False)
        _, _, probs = self.ac.pi(rng1, obs_t, params.pi, False)

        #a_t = jnp.concatenate([a_tm1[:, 1:, :], a_t.reshape(-1, 1, 1)], axis=1)

        # Target Q-values
        q1_targ = self.ac.q1(rng2, obs_t, params.q1_target)
        q2_targ = self.ac.q2(rng3, obs_t, params.q2_target)
        q_targ = jnp.concatenate([q1_targ.reshape(-1,self.act_dim,1), q2_targ.reshape(-1,self.act_dim,1)], axis=2).min(2)
        soft_V = inner1d(probs, q_targ - alpha * jnp.log(probs)) # Eq. 10
        backup = r_t + discount_t * soft_V 
        return jax.lax.stop_gradient(backup)


    @functools.partial(jax.jit, static_argnums=(0,))
    def loss_q1(
        self,
        q1_params: hk.Params,
        params: Params,
        rng: chex.PRNGKey,
        data: Tuple[jnp.ndarray],
        alpha: float = 0.2,
    ):
        obs_tm1, a_tm1, _, _, _ = data
        q1 = self.ac.q1(rng, obs_tm1, q1_params)
        q1 = q1[jnp.arange(q1.shape[0]), a_tm1.astype(int)]
        backup = self.bellman_backup(rng, data, params, alpha)
        loss = ((q1 - backup)**2).mean()
        return loss

    @functools.partial(jax.jit, static_argnums=(0,))
    def loss_q2(
        self,
        q2_params: hk.Params,
        params: Params,
        rng: chex.PRNGKey,
        data: Tuple[jnp.ndarray],
        alpha: float = 0.2,
    ):
        obs_tm1, a_tm1, _, _, _ = data
        q2 = self.ac.q2(rng, obs_tm1, q2_params)
        q2 = q2[jnp.arange(q2.shape[0]), a_tm1.astype(int)]
        backup = self.bellman_backup(rng, data, params, alpha)
        loss = ((q2 - backup)**2).mean()
        return loss
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def loss_pi(
        self,
        pi_params: hk.Params,
        params: Params,
        rng: chex.PRNGKey,
        data: Tuple[jnp.ndarray],
        alpha: float = 0.2,
    ):
        rng1, rng2, rng3 = jax.random.split(rng, num=3)
        obs_tm1, _, _, _, _ = data
        # sample online a_tm1
        _, _, probs = self.ac.pi(rng1, obs_tm1, pi_params, False)

        #a_tm1 = jnp.concatenate([a_history[:, 1:, :], a_tm1.reshape(-1, 1, 1)], axis=1)

        # Compute Q(o,a)
        q1_pi = self.ac.q1(rng2, obs_tm1, params.q1)
        q2_pi = self.ac.q2(rng3, obs_tm1, params.q2)
        q_pi = jnp.concatenate([q1_pi.reshape(-1,self.act_dim,1), q2_pi.reshape(-1,self.act_dim,1)], axis=2).min(2)
        # Entropy-regularized policy loss
        loss_pi = inner1d(probs, alpha * jnp.log(probs) - q_pi).mean() # Eq. 12
        return loss_pi

    @functools.partial(jax.jit, static_argnums=(0,))
    def update(
        self,
        params: Params,
        rng: chex.PRNGKey,
        data: Tuple[jnp.ndarray],
        opt_states: OptStates,
        alpha: float = 0.2,
    ) -> Tuple[Params, OptStates]:
        rng1, rng2, rng3 = jax.random.split(rng, num=3)
        obs_tm1, a_tm1, r_t, discount_t, obs_t = data
        # Update q1
        grads_q1 = jax.grad(self.loss_q1)(params.q1, params, rng1, data, alpha)
        updates_q1, new_opt_state_q1 = self.optimizer.update(grads_q1, opt_states.q1)
        new_params_q1 = optax.apply_updates(params.q1, updates_q1)
        # Update q2
        grads_q2 = jax.grad(self.loss_q2)(params.q2, params, rng2, data, alpha)
        updates_q2, new_opt_state_q2 = self.optimizer.update(grads_q2, opt_states.q2)
        new_params_q2 = optax.apply_updates(params.q2, updates_q2)
        # Update pi
        grads_pi = jax.grad(self.loss_pi)(params.pi, params, rng3, data, alpha)
        updates_pi, new_opt_state_pi = self.optimizer.update(grads_pi, opt_states.pi)
        new_params_pi = optax.apply_updates(params.pi, updates_pi)
        # Update q1_target
        polyak_average = lambda x, y: x*self.polyak + (1 - self.polyak)*y
        #new_params_q1_target = params.q1_target*self.polyak + (1 - self.polyak)*new_params_q1
        new_params_q1_target = jax.tree_map(polyak_average, params.q1_target, new_params_q1)
        # Update q2_target
        #new_params_q2_target = params.q2_target*self.polyak + (1 - self.polyak)*new_params_q2
        new_params_q2_target = jax.tree_map(polyak_average, params.q2_target, new_params_q2)
        return(
            Params(new_params_pi, new_params_q1, new_params_q2, new_params_q1_target, new_params_q2_target),
            OptStates(new_opt_state_pi, new_opt_state_q1, new_opt_state_q2)
        )

    def get_action(
        self,
        rng: chex.PRNGKey,
        params: Params,
        obs: jnp.ndarray, # shape (batch_size, timesteps, features)
        deterministic: bool = False,
    ):
        return self.ac.act(rng, obs, params.pi, deterministic)

    

        

