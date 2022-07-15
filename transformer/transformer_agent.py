import collections
import chex
import distrax
import optax
import functools

#from typing import Optional
from transformers import *

Params = collections.namedtuple("Params", "pi q1 q2 q1_target q2_target")
OptStates = collections.namedtuple("OptStates", "pi q1 q2")
Memory = collections.namedtuple("Memory", "pi q1 q2 q1_target q2_target") # initialize Memory(None, None...)


class GTrXLActor():
    """GTrXL Policy Network"""

    def __init__(
        self, 
        rng: chex.PRNGKey,
        dummy_obs: jnp.ndarray,
        act_dim: int = 3, 
        num_heads: int = 8, 
        key_size: int = 64, 
        num_layers: int = 2, 
        dropout: float = 0.1, 
        hidden_sizes_mlp: Iterable[int] = [],
        dropouta: float = 0.0,
        #name: Optional[str] = None
    ):
        #super().__init__(name)

        def nn_func():
            return apply_GTrXL(dummy_obs.shape[-1], num_heads, key_size, num_layers, dropout, hidden_sizes_mlp, dropouta, act_dim)
        
        self.nn = hk.multi_transform(nn_func)
        self._init_params = self.nn.init(rng, dummy_obs)

    def __call__(
        self,
        rng: chex.PRNGKey,
        obs: jnp.ndarray,
        params: hk.Params,
        deterministic: bool = False,
        memory: Tuple[jnp.ndarray] = None,
    ):
        dict = self.nn.apply.forward(params, rng, obs, memory)
        preds, memory = dict['preds'][:, -1, :], dict['memory']
        categorical_dist = distrax.Categorical(logits=preds)
        actions = jnp.array([0, 1, 2])
        log_probs = categorical_dist.log_prob(actions) 
        if deterministic:
            pi_action = log_probs.argmax()
        else:
            pi_action = categorical_dist.sample(seed=rng)
        log_pi = log_probs[pi_action]
        return pi_action, log_pi, memory

    @property
    def init_params(
        self,
    ):
        return self._init_params

    

class GTrXLCrtitic():
    """GTrXL Critic Network that computes q-values"""

    def __init__(
        self,
        rng: chex.PRNGKey,
        dummy_obs: jnp.ndarray,
        num_heads: int = 8, 
        key_size: int = 64, 
        num_layers: int = 2, 
        dropout: float = 0.1, 
        hidden_sizes_mlp: Iterable[int] = [],
        dropouta: float = 0.0,
    ):
        def nn_func():
            return apply_GTrXL(dummy_obs.shape[-1], num_heads, key_size, num_layers, dropout, hidden_sizes_mlp, dropouta, 1)

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
        act: jnp.ndarray,
        params: hk.Params,
        memory: Tuple[jnp.ndarray] = None,
    ):
        inputs = jnp.concatenate([obs, act], axis=-1)
        dict = self.nn.apply.forward(params, rng, inputs, memory)
        q_values, memory = dict['preds'][:, -1, :], dict['memory']
        return q_values.squeeze(), memory



class GTrXLActorCritic():
    """Class to combine actor and critc networks based on GTrXL"""

    def __init__(
        self,
        rng: chex.PRNGKey,
        dummy_obs_actor: jnp.ndarray,
        dummy_obs_critic: jnp.ndarray,
        obs_dim: int = 1,
        act_dim: int = 3,
        num_heads: int = 8, 
        key_size: int = 64, 
        num_layers: int = 2, 
        dropout: float = 0.1, 
        hidden_sizes_mlp: Iterable[int] = [],
        dropouta: float = 0.0,
    ):
        rng1, rng2, rng3 = jax.random.split(rng, num=3)
        self.pi = GTrXLActor(
            rng = rng1,
            dummy_obs = dummy_obs_actor,
            act_dim = act_dim, 
            num_heads = num_heads, 
            key_size = key_size, 
            num_layers = num_layers, 
            dropout = dropout, 
            hidden_sizes_mlp = hidden_sizes_mlp,
            dropouta = dropout,
        )
        self.q1 = GTrXLCrtitic(
            rng = rng2,
            dummy_obs = dummy_obs_critic,
            num_heads = num_heads, 
            key_size = key_size, 
            num_layers = num_layers, 
            dropout = dropout, 
            hidden_sizes_mlp = hidden_sizes_mlp,
            dropouta = dropout,
        )
        self.q2 = GTrXLCrtitic(
            rng = rng3,
            dummy_obs = dummy_obs_critic,
            num_heads = num_heads, 
            key_size = key_size, 
            num_layers = num_layers, 
            dropout = dropout, 
            hidden_sizes_mlp = hidden_sizes_mlp,
            dropouta = dropout,
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
        memory: Tuple[jnp.ndarray] = None,
    ):
        action, _, memory = self.pi(rng, obs, params, deterministic, memory)
        return action, memory

class GTrXLSAC():
    """Soft-Actor-Critic method based on GTrXL networks"""

    def __init__(
        self,
        rng: chex.PRNGKey,
        dummy_obs_actor: jnp.ndarray,
        dummy_obs_critic: jnp.ndarray,
        obs_dim: int = 1,
        act_dim: int = 3,
        num_heads: int = 8, 
        key_size: int = 64, 
        num_layers: int = 2, 
        dropout: float = 0.1, 
        hidden_sizes_mlp: Iterable[int] = [],
        dropouta: float = 0.0,
        learning_rate: float = 1e-3,
        polyak: float = 0.995,
    ):

        self.ac = GTrXLActorCritic(
            rng = rng,
            dummy_obs_actor = dummy_obs_actor,
            dummy_obs_critic = dummy_obs_critic,
            obs_dim = obs_dim,
            act_dim = act_dim,
            num_heads = num_heads, 
            key_size = key_size, 
            num_layers = num_layers, 
            dropout = dropout, 
            hidden_sizes_mlp = hidden_sizes_mlp,
            dropouta = dropouta,
        )
        self.optimizer = optax.adam(learning_rate)
        self.polyak = polyak

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

    #@functools.partial(jax.jit, static_argnums=(0,))
    def bellman_backup(
        self,
        rng: chex.PRNGKey,
        data: Tuple[jnp.ndarray],
        params: Params,
        memory: Memory,
        alpha: float = 0.2,
    ):
        rng1, rng2, rng3 = jax.random.split(rng, num=3)
        obs_tm1, a_tm1, r_t, discount_t, obs_t = data
        # sample next action
        a_t, logp_a_t, _ = self.ac.pi(rng1, obs_t, params.pi, False, memory.pi)

        a_t = jnp.concatenate([a_tm1[:, 1:, :], a_t.reshape(1, 1, 1)], axis=1)

        # Target Q-values
        q1_targ, _ = self.ac.q1(rng2, obs_t, a_t, params.q1_target, memory.q1_target)
        q2_targ, _ = self.ac.q2(rng3, obs_t, a_t, params.q2_target, memory.q2_target)
        q_targ = jnp.min(jnp.asarray([q1_targ, q2_targ]))
        backup = r_t + discount_t * (q_targ - alpha * logp_a_t) # for last timestep should be only r_t?
        return jax.lax.stop_gradient(backup)


    #@functools.partial(jax.jit, static_argnums=(0,))
    def loss_q1(
        self,
        q1_params: hk.Params,
        params: Params,
        rng: chex.PRNGKey,
        data: Tuple[jnp.ndarray],
        memory: Memory,
        alpha: float = 0.2,
    ):
        obs_tm1, a_tm1, r_t, discount_t, obs_t = data
        q1, _ = self.ac.q1(rng, obs_tm1, a_tm1, q1_params, memory.q1)
        backup = self.bellman_backup(rng, data, params, memory, alpha)
        loss = ((q1 - backup)**2).mean()
        return loss

    #@functools.partial(jax.jit, static_argnums=(0,))
    def loss_q2(
        self,
        q2_params: hk.Params,
        params: Params,
        rng: chex.PRNGKey,
        data: Tuple[jnp.ndarray],
        memory: Memory,
        alpha: float = 0.2,
    ):
        obs_tm1, a_tm1, r_t, discount_t, obs_t = data
        q2, _ = self.ac.q2(rng, obs_tm1, a_tm1, q2_params, memory.q2)
        backup = self.bellman_backup(rng, data, params, memory, alpha)
        loss = ((q2 - backup)**2).mean()
        return loss
    
    #@functools.partial(jax.jit, static_argnums=(0,))
    def loss_pi(
        self,
        pi_params: hk.Params,
        params: Params,
        rng: chex.PRNGKey,
        data: Tuple[jnp.ndarray],
        memory: Memory,
        alpha: float = 0.2,
    ):
        rng1, rng2, rng3 = jax.random.split(rng, num=3)
        obs_tm1, a_history, _, _, _ = data
        # sample online a_tm1
        a_tm1, logp_a_tm1, _ = self.ac.pi(rng1, obs_tm1, pi_params, False, memory.pi)

        a_tm1 = jnp.concatenate([a_history[:, 1:, :], a_tm1.reshape(1, 1, 1)], axis=1)

        # Compute Q(o,a)
        q1_pi, _ = self.ac.q1(rng2, obs_tm1, a_tm1, params.q1, memory.q1)
        q2_pi, _ = self.ac.q2(rng3, obs_tm1, a_tm1, params.q2, memory.q2)
        q_pi = jnp.min(jnp.asarray([q1_pi, q2_pi]))
        # Entropy-regularized policy loss
        loss_pi = (alpha * logp_a_tm1 - q_pi).mean()
        return loss_pi

    #@functools.partial(jax.jit, static_argnums=(0,))
    def update(
        self,
        params: Params,
        rng: chex.PRNGKey,
        data: Tuple[jnp.ndarray],
        memory: Memory,
        opt_states: OptStates,
        alpha: float = 0.2,
    ) -> Tuple[Params, OptStates, Memory]:
        rng1, rng2, rng3 = jax.random.split(rng, num=3)
        obs_tm1, a_tm1, r_t, discount_t, obs_t = data
        # Update q1
        grads_q1 = jax.grad(self.loss_q1)(params.q1, params, rng1, data, memory, alpha)
        _, memory_q1 = self.ac.q1(rng1, obs_tm1, a_tm1, params.q1, memory.q1)
        updates_q1, new_opt_state_q1 = self.optimizer.update(grads_q1, opt_states.q1)
        new_params_q1 = optax.apply_updates(params.q1, updates_q1)
        # Update q2
        grads_q2 = jax.grad(self.loss_q2)(params.q2, params, rng2, data, memory, alpha)
        _, memory_q2 = self.ac.q2(rng2, obs_tm1, a_tm1, params.q2, memory.q2)
        updates_q2, new_opt_state_q2 = self.optimizer.update(grads_q2, opt_states.q2)
        new_params_q2 = optax.apply_updates(params.q2, updates_q2)
        # Update pi
        grads_pi = jax.grad(self.loss_pi)(params.pi, params, rng3, data, memory, alpha)
        _, _, memory_pi = self.ac.pi(rng3, obs_tm1, params.pi, False, memory.pi)
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
            OptStates(new_opt_state_pi, new_opt_state_q1, new_opt_state_q2),
            Memory(memory_pi, memory_q1, memory_q2, memory_q1, memory_q2) # for sure this is wrong
        )

    def get_action(
        self,
        rng: chex.PRNGKey,
        params: Params,
        obs: jnp.ndarray,
        memory_pi_tm1: Tuple[jnp.ndarray] = None,
        deterministic: bool = False,
    ):
        return self.ac.act(rng, obs, params.pi, deterministic, memory_pi_tm1)

    

        

