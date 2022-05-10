from re import A
import haiku as hk
import jax
import numpyro.distributions as dist

from jax import numpy as jnp
from typing import Any, Iterator, Iterable, Generator, Mapping, Tuple, NamedTuple, Sequence, Optional
from jax import lax
from haiku import LSTMState, RNNCore
from haiku._src.recurrent import add_batch, _swap_batch_time, inside_transform
from collections import namedtuple

class BayesianLinear(hk.Module): 
    """
    Bayesian Linear layer based on Bayes-By-Backprop
    """
    def __init__(
        self, 
        hidden_size: int,
        mu_0_prior: jnp.ndarray,
        sigma_0_prior: jnp.ndarray,
        mu_1_prior: jnp.ndarray,
        sigma_1_prior: jnp.ndarray,
        pi_prior: jnp.ndarray,
        with_bias: bool = True,
        name: Optional[str] = None
    ):
        super().__init__(name)

        self.hidden_size = hidden_size
        self.with_bias = with_bias

        self.prior = self.GaussianMixturePrior(
            mu_0=mu_0_prior,
            sigma_0=sigma_0_prior,
            mu_1=mu_1_prior,
            sigma_1=sigma_1_prior,
            pi=pi_prior
        )

        self.log_posterior = None
        self.log_prior = None 

    def sample_weights(
        self,
        mu: jnp.ndarray,
        rho:jnp.ndarray
    ):
        sigma = jax.nn.softplus(rho)*0.05 + 1e-5
        normal_samples = jax.random.normal(hk.next_rng_key(), mu.shape)
        samples = mu + normal_samples * sigma
        return samples

    def GaussianMixturePrior(
        self, 
        mu_0: jnp.ndarray, 
        sigma_0: jnp.ndarray, 
        mu_1: jnp.ndarray, 
        sigma_1: jnp.ndarray, 
        pi: jnp.ndarray,
    ):
        """
        Mixture of two Gaussian distributions as described in Bludell et al., 2015.
        """
        mixing_dist = dist.Categorical(probs=jnp.array([pi, 1-pi]).squeeze())
        component_dist = dist.Normal(loc=jnp.array([mu_0, mu_1]), scale=jnp.array([sigma_0, sigma_1]))
        mixture = dist.MixtureSameFamily(mixing_dist, component_dist)
        return mixture

    def VariationalGaussianPosterior(
        self,
        mu: jnp.ndarray,
        rho: jnp.ndarray
    ):
        sigma = jax.nn.softplus(rho)*0.05 + 1e-5
        normal = dist.Normal(loc=mu, scale=sigma)
        return normal

    def __call__(
        self,
        inputs: jnp.ndarray,
        sample: bool,
        precision: Optional[lax.Precision] = None,
    ) -> Tuple[jnp.ndarray, LSTMState]:

        # Initialize parameters
        input_size = self.input_size = inputs.shape[-1]
        output_size = self.hidden_size
        dtype = inputs.dtype

        std_mu_init = 0.1
        std_rho_init = 1.
        mean_mu_init = 0.
        mean_init = hk.initializers.TruncatedNormal(stddev=std_mu_init, mean=mean_mu_init)
        w_mu = hk.get_parameter(f"w_mu_{self.name}", [input_size, output_size], dtype, init=mean_init)

        rho_init = hk.initializers.TruncatedNormal(stddev=std_rho_init, mean=mean_mu_init)
        w_rho = hk.get_parameter(f"w_rho_{self.name}", [input_size, output_size], dtype, init=rho_init)

        # Sample variational weights
        if sample:
            self.weights = self.sample_weights(mu=w_mu, rho=w_rho)

        # Compute log posterior and log prior for KL divergence
        variational_posterior_w = self.VariationalGaussianPosterior(
            mu=w_mu,
            rho=w_rho
        )
        self.log_posterior = variational_posterior_w.log_prob(self.weights).sum()
        self.log_prior = self.prior.log_prob(self.weights).sum()

        out = jnp.dot(inputs, self.weights, precision=precision)

        if self.with_bias:
            b_init_mu = hk.initializers.TruncatedNormal(stddev=std_mu_init, mean=mean_mu_init)
            b_init_rho = hk.initializers.TruncatedNormal(stddev=std_rho_init, mean=mean_mu_init)
            b_mu = hk.get_parameter(f"b_mu_{self.name}", [output_size], dtype, init=b_init_mu)
            b_mu = jnp.broadcast_to(b_mu, out.shape)
            b_rho = hk.get_parameter(f"b_rho_{self.name}", [output_size], dtype, init=b_init_rho)
            b_rho = jnp.broadcast_to(b_rho, out.shape)
            if sample:
                self.bias = self.sample_weights(mu=b_mu, rho=b_rho)
            out = out + self.bias
            variational_posterior_b = self.VariationalGaussianPosterior(
                mu=b_mu,
                rho=b_rho
            )
            self.log_posterior += variational_posterior_b.log_prob(self.bias).sum()
            self.log_prior += self.prior.log_prob(self.bias).sum()

        return out

class BayesianLSTM(RNNCore):
    """
    Bayesian LSTM layer based on Bayes-By-Backprop
    """
    def __init__(
        self, 
        hidden_size: int,
        mu_0_prior: jnp.ndarray,
        sigma_0_prior: jnp.ndarray,
        mu_1_prior: jnp.ndarray,
        sigma_1_prior: jnp.ndarray,
        pi_prior: jnp.ndarray,
        with_bias: bool = True,
        name: Optional[str] = None
    ):
        super().__init__(name)

        self.hidden_size = hidden_size
        self.with_bias = with_bias

        self.prior = self.GaussianMixturePrior(
            mu_0=mu_0_prior,
            sigma_0=sigma_0_prior,
            mu_1=mu_1_prior,
            sigma_1=sigma_1_prior,
            pi=pi_prior
        )

        self.log_posterior = None
        self.log_prior = None 

    def sample_weights(
        self,
        mu: jnp.ndarray,
        rho:jnp.ndarray
    ):
        sigma = jax.nn.softplus(rho)*0.05 + 1e-5
        normal_samples = jax.random.normal(hk.next_rng_key(), mu.shape)
        samples = mu + normal_samples * sigma
        return samples

    def GaussianMixturePrior(
        self, 
        mu_0: jnp.ndarray, 
        sigma_0: jnp.ndarray, 
        mu_1: jnp.ndarray, 
        sigma_1: jnp.ndarray, 
        pi: jnp.ndarray,
    ):
        """
        Mixture of two Gaussian distributions as described in Bludell et al., 2015.
        """
        mixing_dist = dist.Categorical(probs=jnp.array([pi, 1-pi]).squeeze())
        component_dist = dist.Normal(loc=jnp.array([mu_0, mu_1]), scale=jnp.array([sigma_0, sigma_1]))
        mixture = dist.MixtureSameFamily(mixing_dist, component_dist)
        return mixture

    def VariationalGaussianPosterior(
        self,
        mu: jnp.ndarray,
        rho: jnp.ndarray
    ):
        sigma = jax.nn.softplus(rho)*0.05 + 1e-5
        normal = dist.Normal(loc=mu, scale=sigma)
        return normal

    def __call__(
        self,
        inputs: jnp.ndarray,
        prev_state: LSTMState,
        sample: bool,
        precision: Optional[lax.Precision] = None,
    ) -> Tuple[jnp.ndarray, LSTMState]:

        x_and_h = jnp.concatenate([inputs, prev_state.hidden], axis=-1)

        # Initialize parameters
        input_size = self.input_size = x_and_h.shape[-1]
        output_size = 4 * self.hidden_size
        dtype = x_and_h.dtype

        std_mu_init = 0.1
        std_rho_init = 1.
        mean_mu_init = 0.

        mean_init = hk.initializers.TruncatedNormal(stddev=std_mu_init, mean=mean_mu_init)
        w_mu = hk.get_parameter(f"w_mu_{self.name}", [input_size, output_size], dtype, init=mean_init)

        rho_init = hk.initializers.TruncatedNormal(stddev=std_rho_init, mean=mean_mu_init)
        w_rho = hk.get_parameter(f"w_rho_{self.name}", [input_size, output_size], dtype, init=rho_init)

        # Sample variational weights
        if sample:
            self.weights = self.sample_weights(mu=w_mu, rho=w_rho)

        # Compute log posterior and log prior for KL divergence
        variational_posterior_w = self.VariationalGaussianPosterior(
            mu=w_mu,
            rho=w_rho
        )
        self.log_posterior = variational_posterior_w.log_prob(self.weights).sum()
        self.log_prior = self.prior.log_prob(self.weights).sum()

        gated = jnp.dot(x_and_h, self.weights, precision=precision)

        if self.with_bias:
            b_init_mu = hk.initializers.TruncatedNormal(stddev=std_mu_init, mean=mean_mu_init)
            b_init_rho = hk.initializers.TruncatedNormal(stddev=std_rho_init, mean=mean_mu_init)
            b_mu = hk.get_parameter(f"b_mu_{self.name}", [output_size], dtype, init=b_init_mu)
            b_mu = jnp.broadcast_to(b_mu, gated.shape)
            b_rho = hk.get_parameter(f"b_rho_{self.name}", [output_size], dtype, init=b_init_rho)
            b_rho = jnp.broadcast_to(b_rho, gated.shape)
            if sample:
                self.bias = self.sample_weights(mu=b_mu, rho=b_rho)
            gated = gated + self.bias
            variational_posterior_b = self.VariationalGaussianPosterior(
                mu=b_mu,
                rho=b_rho
            )
            self.log_posterior += variational_posterior_b.log_prob(self.bias).sum()
            self.log_prior += self.prior.log_prob(self.bias).sum()

        # Compute hidden states as in LSTM
        i, g, f, o = jnp.split(gated, indices_or_sections=4, axis=-1)
        f = jax.nn.sigmoid(f + 1)  # Forget bias, as in sonnet.
        c = f * prev_state.cell + jax.nn.sigmoid(i) * jnp.tanh(g)
        h = jax.nn.sigmoid(o) * jnp.tanh(c)
        return h, LSTMState(h, c)

    def initial_state(self, batch_size: Optional[int]) -> LSTMState:
        state = LSTMState(hidden=jnp.zeros([self.hidden_size]),
                            cell=jnp.zeros([self.hidden_size]))
        if batch_size is not None:
            state = add_batch(state, batch_size)
        return state

class BayesianLSTM_Sharpening(BayesianLSTM):
    """
    Bayesian LSTM layer with posterior sharpening (Fortunato et al., 2017).
    """
    def __init__(
        self, 
        hidden_size: int, 
        mu_0_prior: jnp.ndarray, 
        sigma_0_prior: jnp.ndarray, 
        mu_1_prior: jnp.ndarray, 
        sigma_1_prior: jnp.ndarray, 
        pi_prior: jnp.ndarray, 
        sigma_0_sharpen: jnp.ndarray,
        with_bias: bool = True, 
        name: Optional[str] = None
    ):
        super().__init__(hidden_size, mu_0_prior, sigma_0_prior, mu_1_prior, sigma_1_prior, pi_prior, with_bias, name)

        self.sigma_0_sharpen = sigma_0_sharpen
        self.sharpen = True

    def sharpening(
        self, 
    ):
        raise NotImplementedError

class BayesianDeepRNN(RNNCore):
    """
    Underlying implementation of BayesianDeepRNN (without skip connections)
    """


    def __init__(
        self,
        layers: Sequence[Any],
        name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.layers = layers

    def __call__(self, inputs, state, sample):
        current_inputs = inputs
        next_states = []
        outputs = []
        state_idx = 0
        log_posterior = 0
        log_prior = 0
        for layer in self.layers:
            if isinstance(layer, RNNCore):
                current_inputs, next_state = layer(current_inputs, state[state_idx], sample)
                outputs.append(current_inputs)
                next_states.append(next_state)
                state_idx += 1
                log_posterior += layer.log_posterior
                log_prior += layer.log_prior
            elif isinstance(layer, hk.Module):
                if layer.name == 'bayesian_out_mean_linear_layer':
                    out_mean = layer(current_inputs, sample)
                    log_posterior += layer.log_posterior
                    log_prior += layer.log_prior
                elif layer.name == 'bayesian_out_rho_linear_layer':
                    out_rho = layer(current_inputs, sample)
                    log_posterior += layer.log_posterior
                    log_prior += layer.log_prior
                else:
                    raise ValueError
            else:
                current_inputs = layer(current_inputs)

        return out_mean, out_rho, tuple(next_states), log_posterior, log_prior

    def initial_state(self, batch_size: Optional[int]):
        return tuple(
            layer.initial_state(batch_size)
            for layer in self.layers
            if isinstance(layer, RNNCore))


class BayesianDeepLSTM():
    """
    Bayesian NN based on BayesianLSTM layer
    """
    def __init__(
        self,
        hidden_units: Sequence[int],
        output_units: int,
        mu_0_prior: jnp.ndarray,
        sigma_0_prior: jnp.ndarray,
        mu_1_prior: jnp.ndarray,
        sigma_1_prior: jnp.ndarray,
        pi_prior: jnp.ndarray,
        with_bias: bool = True,
    ):
        self.hidden_units = hidden_units
        self.output_units = output_units

        self.mu_0_prior = mu_0_prior
        self.sigma_0_prior = sigma_0_prior
        self.mu_1_prior = mu_1_prior
        self.sigma_1_prior = sigma_1_prior
        self.pi_prior = pi_prior
        self.with_bias = with_bias
        
        self.log_posterior = None
        self.log_prior = None

    def network(self):
        layers = [lambda x : x.reshape(-1, 1)]
        for idx, num_units in enumerate(self.hidden_units):
            layers.append(BayesianLSTM(
                hidden_size=num_units,
                mu_0_prior=self.mu_0_prior,
                sigma_0_prior=self.sigma_0_prior,
                mu_1_prior=self.mu_1_prior,
                sigma_1_prior=self.sigma_1_prior,
                pi_prior=self.pi_prior,
                with_bias=self.with_bias,
                name=f"bayesian_lstm_layer_{idx}"
            ))
            if idx != len(self.hidden_units)-1: 
                layers.append(jax.nn.relu)
        layers.append(BayesianLinear(
            hidden_size=self.output_units,
            mu_0_prior=self.mu_0_prior,
            sigma_0_prior=self.sigma_0_prior,
            mu_1_prior=self.mu_1_prior,
            sigma_1_prior=self.sigma_1_prior,
            pi_prior=self.pi_prior,
            with_bias=self.with_bias,
            name=f"bayesian_out_mean_linear_layer"
        ))
        layers.append(BayesianLinear(
            hidden_size=self.output_units,
            mu_0_prior=self.mu_0_prior,
            sigma_0_prior=self.sigma_0_prior,
            mu_1_prior=self.mu_1_prior,
            sigma_1_prior=self.sigma_1_prior,
            pi_prior=self.pi_prior,
            with_bias=self.with_bias,
            name=f"bayesian_out_rho_linear_layer"
        ))
        model = BayesianDeepRNN(layers)
        return model

    def static_unroll(self, obs:jnp.ndarray):
        net = self.network()
        batch_size = obs.shape[0]
        initial_state = net.initial_state(batch_size)
        means, rhos, states, log_posterior, log_prior = bayesian_static_unroll(net, obs, initial_state)
        return means, rhos, states, log_posterior, log_prior

    def forward(self, obs:jnp.ndarray):
        means, rhos, _, log_posterior, log_prior = self.static_unroll(obs)
        sigmas = jax.nn.softplus(rhos[:, -1, :])*0.05 + 1e-5
        return means[:, -1, :], sigmas, log_posterior, log_prior

    def predict_posterior(self, obs:jnp.ndarray, n_samples: int = 100):
        """Predict a full posterior distribution. Implement a method that parallelize over n_samples"""
        ### TODO: current implementation is wrong because it will not produce different samples because of rng_key. 
        obs = jnp.stack([obs]*n_samples)
        post_means, post_sigmas, _, _ = jax.vmap(self.forward, in_axes=(0,), out_axes=0)(obs)
        return post_means, post_sigmas

def bayesian_static_unroll(core, input_sequence, initial_state):
    """Performs a static unroll of an RNN.
    An *unroll* corresponds to calling the core on each element of the
    input sequence in a loop, carrying the state through::
        state = initial_state
        for t in range(len(input_sequence)):
            outputs, state = core(input_sequence[t], state)
    A *static* unroll replaces a loop with its body repeated multiple
    times when executed inside :func:`jax.jit`::
        state = initial_state
        outputs0, state = core(input_sequence[0], state)
        outputs1, state = core(input_sequence[1], state)
        outputs2, state = core(input_sequence[2], state)
        ...
    See :func:`dynamic_unroll` for a loop-preserving unroll function.
    Args:
        core: An :class:`RNNCore` to unroll.
        input_sequence: An arbitrarily nested structure of tensors of shape
        ``[B, T, ...]`` where ``T`` is the number of time steps.
        initial_state: An initial state of the given core.
    Returns:
        A tuple with two elements:
        * **output_sequence** - An arbitrarily nested structure of tensors``[B, T, ...]``.
        * **final_state** - Core state at time step ``T``.
    """
    means_sequence = []
    rhos_sequence = []
    log_posterior_sequence = []
    log_prior_sequence = []
    time_axis = 1 
    num_steps = jax.tree_leaves(input_sequence)[0].shape[time_axis]
    state = initial_state
    for t in range(num_steps):
        inputs = jax.tree_map(lambda x, _t=t: x[:, _t], input_sequence)
        if t==0: # sample fresh parameters for a new mini-batch
            means, rhos, state, log_posterior, log_prior = core(inputs, state, sample=True)
        else:
            means, rhos, state, log_posterior, log_prior = core(inputs, state, sample=False)
        means_sequence.append(means)
        rhos_sequence.append(rhos)
        log_posterior_sequence.append(log_posterior)
        log_prior_sequence.append(log_prior)

    # Stack outputs along the time axis.
    means_sequence = jax.tree_multimap(
        lambda *args: jnp.stack(args, axis=time_axis),
        *means_sequence)
    rhos_sequence = jax.tree_multimap(
        lambda *args: jnp.stack(args, axis=time_axis),
        *rhos_sequence)
    return means_sequence, rhos_sequence, state, jnp.array(log_posterior_sequence)[-1], jnp.array(log_prior_sequence)[-1]


BayesianDeepLSTMApply = namedtuple(
    'BayesianDeepLSTMApply',
    ['static_unroll', 'forward', 'predict_posterior']
)

def apply_BayesianDeepLSTM(
    hidden_units, 
    output_units,
    mu_0_prior=jnp.array(0.),
    sigma_0_prior=jnp.array(0.368),
    mu_1_prior=jnp.array(0.),
    sigma_1_prior=jnp.array(0.00091),
    pi_prior=jnp.array(0.5),
    with_bias=True
):
    bayesian_lstm = BayesianDeepLSTM(
        hidden_units=hidden_units, 
        output_units=output_units,
        mu_0_prior=mu_0_prior,
        sigma_0_prior=sigma_0_prior,
        mu_1_prior=mu_1_prior,
        sigma_1_prior=sigma_1_prior,
        pi_prior=pi_prior,
        with_bias=with_bias
    )

    def init(x):
        return bayesian_lstm.forward(x)

    return init, BayesianDeepLSTMApply(
        bayesian_lstm.static_unroll, 
        bayesian_lstm.forward,
        bayesian_lstm.predict_posterior
    )