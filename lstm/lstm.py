import haiku as hk
import jax

from jax import numpy as jnp
from typing import Any, Iterator, Iterable, Generator, Mapping, Tuple, NamedTuple, Sequence
from collections import namedtuple


class DeepLSTM():

  def __init__(
      self,
      hidden_units: Sequence[int],
      output_units: int,   
      ):
    """LSTM nn architecture"""

    self.hidden_units = hidden_units
    self.output_units = output_units

  def network(self):
      layers = []
      for idx, num_units in enumerate(self.hidden_units):
        layers.append(hk.LSTM(num_units, name=f"lstm_layer_{idx}"))
        if idx != len(self.hidden_units)-1: 
          layers.append(jax.nn.relu)
      layers.append(hk.Linear(self.output_units, name=f"out_layer"))
      #layers.append(hk.nets.MLP([self.output_units])),
      model = hk.DeepRNN(layers)
      return model

  def dynamic_unroll(self, obs:jnp.ndarray):
      net = self.network()
      batch_size = obs.shape[0]
      initial_state = net.initial_state(batch_size)
      outs, states = hk.static_unroll(net, obs, initial_state, time_major=False)
      return outs, states

  def forward(self, obs:jnp.ndarray):
        outs, _ = self.dynamic_unroll(obs)
        return outs[:, -1, :]

DeepLSTMApply = namedtuple(
    'DeepLSTMApply',
    ['dynamic_unroll', 'forward']
)

def apply_DeepLSTM(hidden_units, output_units):
  lstm = DeepLSTM(
    hidden_units=hidden_units, 
    output_units=output_units
  )
  def init(x):
    return lstm.dynamic_unroll(x)

  return init, DeepLSTMApply(lstm.dynamic_unroll, lstm.forward)
