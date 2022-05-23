"""
Implementation of stable transformer XL https://arxiv.org/pdf/1910.06764.pdf. Some of this code comes from
the torch implementations: 
- https://github.com/alantess/gtrxl-torch/blob/main/gtrxl_torch/gtrxl_torch.py.
- https://github.com/dhruvramani/Transformers-RL
"""

import haiku as hk
import jax

import numpy as np
from jax import numpy as jnp
from typing import Any, Iterator, Iterable, Generator, Mapping, Tuple, NamedTuple, Sequence
from collections import namedtuple

def twos(shape, dtype):
    zeros = jnp.zeros(shape, dtype)
    return zeros + 2

class GRUGate(hk.Module):
    """
    Gru gating layer as defined in stable tranformers XL paper.
    """
    def __init__(
        self,
        input_dimension: int
    ):
        super().__init__()
        self.input_d = input_dimension
        self.Wr = hk.Linear(input_dimension, with_bias=False)
        self.Ur = hk.Linear(input_dimension, with_bias=False)
        self.Wz = hk.Linear(input_dimension, with_bias=False) 
        self.Uz = hk.Linear(input_dimension, with_bias=False)
        self.Wg = hk.Linear(input_dimension, with_bias=False) 
        self.Ug = hk.Linear(input_dimension, with_bias=False)
        self.bg = hk.get_parameter(f"bg", [input_dimension], jnp.float32, init=twos)

    def __call__(
        self,
        x: jnp.ndarray,
        y: jnp.ndarray
    ):
        r = jax.nn.sigmoid(self.Wr(y) + self.Ur(x))
        z = jax.nn.sigmoid(self.Wz(y) + self.Uz(x) - self.bg)
        h = jax.nn.tanh(self.Wg(y) + self.Ug(r*x))
        g = (1-z)*x + z*h
        return g

class PositionalEncoding(hk.Module):
    """
    Stolen from https://github.com/alantess/gtrxl-torch/blob/main/gtrxl_torch/gtrxl_torch.py
    """
    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super(PositionalEncoding, self).__init__()
        self.dropout = dropout
        pe = np.zeros((max_len, d_model))
        position = np.expand_dims(np.arange(0, max_len),1)
        div_term = np.exp(
            np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        pe = np.expand_dims(pe, 0)
        pe = np.swapaxes(pe, 0, 1)
        #pe = pe.unsqueeze(0).transpose(0, 1)
        self.pe = jnp.asarray(pe)

    def __call__(self, x):
        x = x + self.pe[:x.shape[0], :]
        return hk.dropout(rng=hk.next_rng_key(), rate= self.dropout, x=x)

class GTrXLCore(hk.Module):
    """ 
    GTrXL layer block
    """
    def __init__(
        self,
        model_dimension: int = 1,
        num_heads: int = 8, 
        key_size: int = 64,
        dropout: float = 0.1,
        max_len: int = 1024,
        hidden_sizes_mlp: Iterable[int] = [],
    ):
        super().__init__()
        #model_dimension = num_heads*key_size
        self.positional_encoding = PositionalEncoding(d_model=model_dimension, dropout=dropout, max_len=max_len)
        self.layernorm1 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True) ### axis = [-ax, -ax-1, ...]
        self.layernorm2 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        self.mha = hk.MultiHeadAttention(num_heads=num_heads, key_size=key_size, w_init_scale=1)
        self.gated_mha = GRUGate(input_dimension=model_dimension)
        self.mlp = hk.nets.MLP(hidden_sizes_mlp+[model_dimension])
        self.gated_mlp = GRUGate(input_dimension=model_dimension)


    def __call__(
        self,
        x
    ):
        x_pos = self.positional_encoding(x)
        x_norm = self.layernorm1(x_pos)
        y_bar = self.mha(x_norm, x_norm, x_norm)
        y = self.gated_mha(x, jax.nn.relu(y_bar))
        e_bar = self.mlp(self.layernorm2(y))
        e = self.gated_mlp(y, jax.nn.relu(e_bar))
        return e

class GTrXL():
    """ 
    GTrXL decoder
    """
    def __init__(
        self,
        model_dimension: int = 1,
        num_heads: int = 8, 
        key_size: int = 64,
        num_layers: int = 12,
        dropout: float = 0.1,
        max_len: int = 50,
        hidden_sizes_mlp: Iterable[int] = [],
        output_size: int = 1,
    ):
        self.model_dimension = model_dimension
        self.num_heads = num_heads 
        self.key_size = key_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.max_len  = max_len 
        self.hidden_sizes_mlp = hidden_sizes_mlp
        self.output_size = output_size

    def forward(
        self,
        x
    ):
        for layer in range(self.num_layers):
            x = GTrXLCore(model_dimension=self.model_dimension, num_heads=self.num_heads, key_size=self.key_size, dropout=self.dropout, 
                          max_len=self.max_len, hidden_sizes_mlp=self.hidden_sizes_mlp)(x)
        x = hk.Linear(output_size=self.output_size)(x) ### I haven't figured out yet if this is needed or one can just set model_dimension=output_size

        return x[:, -1, :]

GTrXLApply = namedtuple(
    'GTrXLApply',
    ['forward']
)

def apply_GTrXL(model_dimension, num_heads, key_size, num_layers, dropout, max_len, hidden_sizes_mlp, output_size):
  transformer = GTrXL(model_dimension, num_heads, key_size, num_layers, dropout, max_len, hidden_sizes_mlp, output_size)
  def init(x):
    return transformer.forward(x)

  return init, GTrXLApply(transformer.forward)
