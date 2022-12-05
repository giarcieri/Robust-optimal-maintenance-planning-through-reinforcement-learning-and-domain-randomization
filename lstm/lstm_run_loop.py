import time
import pickle
import os, sys

from .lstm_agent import *
dir2 = os.path.abspath('')
dir1 = os.path.dirname(dir2)
if not dir1 in sys.path: sys.path.append(dir1)
from environment.fractal_env_jax import *

class ReplayBufferPO(object):
    """A replay buffer for POMDPs that stores time-series of observations"""

    def __init__(self, capacity, episode_horizon=50, window_length=5, gamma=0.995):
        self.obs_buffer = jnp.zeros((capacity, episode_horizon))
        self.action_buffer = jnp.zeros((capacity, episode_horizon))
        self.reward_buffer = jnp.zeros((capacity, episode_horizon))
        self.capacity = capacity
        self.episode_horizon = episode_horizon
        self.window_length = window_length
        self.gamma = gamma

    def push(self, obs, action, reward, episode, step):
        self.size_buffer = min(episode+1, self.capacity)
        episode = episode % self.capacity
        self.obs_buffer = self.obs_buffer.at[episode, step].set(obs.squeeze())
        self.action_buffer = self.action_buffer.at[episode, step].set(action.squeeze())
        self.reward_buffer = self.reward_buffer.at[episode, step].set(reward.squeeze())

    def sample(self, idx): 
        obs_trajectory = self.obs_buffer[idx]
        action_trajectory = self.action_buffer[idx]
        reward_trajectory = self.reward_buffer[idx]
        sub_windows = (
            jnp.expand_dims(jnp.arange(self.window_length), 0) +
            jnp.expand_dims(jnp.arange(self.episode_horizon + 1 - self.window_length), 0).T
            #jnp.expand_dims(jnp.arange(self.episode_horizon), 0).T
        )
        # observations
        #nans = jnp.full((self.window_length - 1), jnp.NINF)
        #obs_trajectory = jnp.concatenate([nans, obs_trajectory])
        timesteps = jnp.expand_dims(jnp.arange(self.window_length - 1, self.episode_horizon), 1)

        obs_sliding_window = obs_trajectory[sub_windows] # shape (episode_horizon + 1 - window_length, window_length)
        obs_tm1_trajectory = obs_sliding_window[:-1, :] # shape (episode_horizon - window_length, window_length)
        obs_t_trajectory = obs_sliding_window[1:, :] # shape (episode_horizon - window_length, window_length)

        obs_tm1_trajectory = jnp.concatenate([obs_tm1_trajectory, timesteps[:-1]], axis=1)
        obs_t_trajectory = jnp.concatenate([obs_t_trajectory, timesteps[1:]], axis=1)

        # actions 
        #action_sliding_window = action_trajectory[sub_windows]
        #a_tm1_trajectory = action_sliding_window[:-1, :]
        a_tm1_trajectory = action_trajectory[self.window_length-1:-1]

        # rewards
        #rewards_sliding_window = reward_trajectory[sub_windows]
        #r_t_trajectory = rewards_sliding_window[1:, -1]
        r_t_trajectory = reward_trajectory[self.window_length:]

        # discounts
        discount_t_trajectory = jnp.zeros(r_t_trajectory.shape) + self.gamma
        return (obs_tm1_trajectory, a_tm1_trajectory, r_t_trajectory, discount_t_trajectory, obs_t_trajectory)

    def batch_sample(self, idxs):
        obs_tm1_batch, a_tm1_batch, r_t_batch, discount_t_batch, obs_t_batch = jax.vmap(self.sample)(idxs)
        obs_tm1_batch = obs_tm1_batch.reshape(-1, self.window_length + 1, 1)
        a_tm1_batch = a_tm1_batch.reshape(-1,)
        r_t_batch = r_t_batch.reshape(-1,)
        discount_t_batch = discount_t_batch.reshape(-1,)
        obs_t_batch = obs_t_batch.reshape(-1, self.window_length + 1, 1)
        return (obs_tm1_batch, a_tm1_batch, r_t_batch, discount_t_batch, obs_t_batch)



def run_loop(
    seed: int,
    reward_matrix: jnp.ndarray,
    domain_randomization: bool,
    trace: dict,
    window_length: int,
    use_action_history: bool = False,
    replay_size: int = int(1e6),
    train_episodes: int = int(1e2),
    step_per_episode: int = 50,
    gamma: float = 0.99,
    update_every: int = 1,
    update_iterations: int = 10,
    gradient_descent_epochs: int = 10,
    update_after: int = 1,
    batch_size: int = 2,
    test_episodes: int = int(1e2),
    domain_randomization_test: bool = True,
    obs_dim: int = 1,
    act_dim: int = 3,
    hidden_sizes: Iterable[int] = [100, 100],
    learning_rate: float = 1e-3,
    polyak: float = 0.995,
    alpha: float = 0.2,
    save_rewards: bool = True,
    save_model: bool = True,
    gridsearch: bool = False,

):
    rng = hk.PRNGSequence(seed)

    # Environment
    env = FractalEnv(reward_matrix=reward_matrix)
    dummy_obs = []
    #dummy_action = []
    for _ in range(window_length):
        dummy_obs.append(env.observation_space().sample(next(rng)))
        #dummy_action.append(env.action_space().sample(next(rng)))
    dummy_obs = jnp.asarray(dummy_obs).reshape((1, window_length, 1))
    #dummy_action = jnp.asarray(dummy_action).reshape((1, window_length, 1))
    #if use_action_history:
    #    raise(NotImplementedError)
    #else:
    #    dummy_obs_critic = jnp.concatenate([dummy_obs, dummy_action], axis=-1)

    # add timestep
    dummy_obs = jnp.concatenate([dummy_obs, jnp.array([1.]).reshape(1,1,1)], axis=1)

    # Replay Buffer
    buffer = ReplayBufferPO(
        capacity=replay_size//step_per_episode, 
        episode_horizon=step_per_episode, 
        window_length=window_length, 
        gamma=gamma
    )

    # Agent
    agent = LSTMSAC(
        rng=next(rng),
        dummy_obs_actor = dummy_obs,
        dummy_obs_critic = dummy_obs,
        obs_dim = obs_dim,
        act_dim = act_dim,
        hidden_sizes = hidden_sizes,
        learning_rate = learning_rate,
        polyak = polyak,
    )
    agent_params = agent.init_params()
    opt_states = agent.init_opt_state(agent_params)

    # Train Loop
    start_time = time.time()
    tot_train_ep_returns = []
    #with open("lstm/logs.txt", "a") as f:
    #    f.write(f"Start training loop\n")
    for train_episode in range(train_episodes):
        train_ep_return = 0
        if domain_randomization:
            env_params = sample_params(next(rng), trace)
        else:
            env_params = sample_mean_params(trace)
        obs_tm1, hs_tm1 = env.reset(next(rng), env_params)
        # start at hs 0 to initially limit variance
        #hs_tm1 = jnp.array(0)
        obs_tm1_full_history = jnp.full((1, step_per_episode, 1), jnp.NINF)
        for step in range(step_per_episode):
            obs_tm1_full_history = obs_tm1_full_history.at[:, step, :].set(obs_tm1)
            if step < window_length:
                obs_tm1_history = obs_tm1_full_history[:, :step+1, :]
                obs_tm1_history = jnp.concatenate([jnp.zeros((1, window_length-(step+1), 1)), obs_tm1_history], axis=1)
            else:
                obs_tm1_history = obs_tm1_full_history[:, step-window_length+1:step+1, :]

            # add timestep
            obs_tm1_history = jnp.concatenate([obs_tm1_history, jnp.array([step]).reshape(1,1,1)], axis=1)

            a_tm1 = agent.get_action(
                rng = next(rng),
                params = agent_params,
                obs = obs_tm1_history,
                deterministic = False,
            )
            obs_t, hs_t, r_t, _, _ = env.step(next(rng), hs_tm1, obs_tm1, a_tm1.squeeze(), env_params)
            train_ep_return += r_t
            buffer.push(obs=obs_tm1, action=a_tm1, reward=r_t, episode=train_episode, step=step)
            #print(f'Episode {train_episode} Step {step}: o_tm1 {obs_tm1.round(2)}, hs_tm1 {hs_tm1}, a_tm1 {a_tm1}, r_t {r_t}')
            obs_tm1, hs_tm1 = obs_t, hs_t
        print(f'Episode {train_episode} total return {train_ep_return}')
        # Update
        if train_episode >= update_after and train_episode % update_every == 0:
            #with open("lstm/logs.txt", "a") as f:
            #    f.write(f"Update\n")
            for _ in range(update_every*update_iterations):
                idxs = jax.random.randint(next(rng), shape=(batch_size,), minval=0, maxval=buffer.size_buffer)
                batch = buffer.batch_sample(idxs)
                for _ in range(gradient_descent_epochs):
                    agent_params, opt_states = agent.update(
                        params = agent_params,
                        rng = next(rng),
                        data = batch,
                        opt_states = opt_states,
                        alpha = alpha,
                    )
        # Collect episode return
        tot_train_ep_returns.append(train_ep_return)
    train_time = time.time()-start_time
    print(f"Training time: {round(train_time/3600, 1)}h")
    # Save train episode returns 
    if save_rewards:
        file = 'lstm/rewards/train_rewards_LSTM_' + 'seed'+str(seed) + '_' + time.strftime("%d-%m-%Y")+ '.pickle'
        with open(file, "wb") as fp:
            pickle.dump(tot_train_ep_returns, fp)
    # Test
    start_time = time.time() 
    tot_test_ep_returns = []
    #with open("lstm/logs.txt", "a") as f:
    #    f.write(f"Starting test\n")
    for test_episode in range(test_episodes):
        test_ep_return = 0
        if domain_randomization_test:
            env_params = sample_params(next(rng), trace)
        else:
            env_params = sample_mean_params(trace)
        obs, hs = env.reset(next(rng), env_params)
        # start at hs 0 to initially limit variance
        #hs = jnp.array(0)
        obs_full_history = jnp.full((1, step_per_episode, 1), jnp.NINF)
        for step in range(step_per_episode):
            obs_full_history = obs_full_history.at[:, step, :].set(obs)
            if step < window_length:
                obs_history = obs_full_history[:, :step+1, :]
                obs_history = jnp.concatenate([jnp.zeros((1, window_length-(step+1), 1)), obs_history], axis=1)
            else:
                obs_history = obs_full_history[:, step-window_length+1:step+1, :]

            # add timestep
            obs_history = jnp.concatenate([obs_history, jnp.array([step]).reshape(1,1,1)], axis=1)

            a = agent.get_action(
                rng = next(rng),
                params = agent_params,
                obs = obs_history,
                deterministic = True,
            )
            print(f'Episode {test_episode} Step {step}: o_tm1 {obs.round(2)}, hs_tm1 {hs}, a_tm1 {a}')
            obs, hs, r, _, _ = env.step(next(rng), hs, obs, a.squeeze(), env_params)
            test_ep_return += r
            print(f'Episode {test_episode} Step {step}: r_t {r}')
        #print(f'Episode {test_episode} total return {test_ep_return}')
        # Collect test episode return
        tot_test_ep_returns.append(test_ep_return)
    test_time = time.time()-start_time
    print(f"Testing time: {round(test_time/3600, 1)}h")
    # Save train episode returns 
    if save_rewards:
        file = 'lstm/rewards/test_rewards_LSTM_' + 'seed'+str(seed) + '_' + time.strftime("%d-%m-%Y")+ '.pickle'
        with open(file, "wb") as fp:
            pickle.dump(tot_test_ep_returns, fp)
    # Save model
    if save_model:
        params_memory = {'params': agent_params}
        file = 'lstm/saved_models/model_LSTM_' + 'seed'+str(seed) + '_' + time.strftime("%d-%m-%Y")+ '.pickle'
        with open(file, "wb") as fp:
            pickle.dump(params_memory, fp)
    if gridsearch:
        with open("lstm/gridsearch_results/gridsearch_results.txt", "a") as f:
            f.write(f"seed {seed} train_episodes {train_episodes} update_iterations {update_iterations} gradient_descent_epochs {gradient_descent_epochs} hidden_sizes {hidden_sizes} learning_rate {learning_rate} alpha {alpha} polyak {polyak} replay_size {replay_size}: mean {int(jnp.asarray(tot_test_ep_returns).mean())} std {int(jnp.asarray(tot_test_ep_returns).std())}\n")

        
