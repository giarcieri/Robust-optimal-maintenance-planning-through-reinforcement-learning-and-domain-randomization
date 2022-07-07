"""
Implementation of stable transformer XL https://arxiv.org/pdf/1910.06764.pdf. Some of this code comes from
the torch implementation: 
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
    def __init__(self, dim):
        super(PositionalEncoding, self).__init__()

        self.dim = dim

        inv_freq = 1 / (10000 ** (jnp.arange(0.0, dim, 2.0) / dim))
        self.inv_freq = inv_freq

    def __call__(self, pos_seq):
        sinusoid_inp = jnp.outer(pos_seq, self.inv_freq)
        pos_emb = jnp.concatenate([jnp.sin(sinusoid_inp), jnp.cos(sinusoid_inp)], axis=-1)
        return pos_emb[None, :, :]

class MultiHeadAttentionXL(hk.Module):
    def __init__(
        self, 
        d_input: int, 
        d_head_inner: int, 
        n_heads: int = 8, 
        dropout: float = 0.1, 
        dropouta: float = 0.0
    ):
        super(MultiHeadAttentionXL, self).__init__()

        self.d_input = d_input
        self.d_head_inner = d_head_inner
        self.n_heads = n_heads

        # Linear transformation for keys & values for all heads at once for efficiency.
        self.linear_kv = hk.Linear((d_head_inner * n_heads * 2), with_bias=False)
        # for queries (will not be concatenated with memorized states so separate).
        self.linear_q = hk.Linear(d_head_inner * n_heads, with_bias=False)

        # for positional embeddings.
        self.linear_p = hk.Linear(d_head_inner * n_heads, with_bias=False)
        self.scale = 1 / (d_head_inner ** 0.5)  # for scaled dot product attention
        self.dropa = dropouta

        self.lout = hk.Linear(d_input, with_bias=False)
        self.dropo = dropout

    def _rel_shift(self, x):
        # x shape: [B x curr x curr+prev x n_heads]
        zero_pad = jnp.zeros((*x.shape[:2], 1, x.shape[-1]), dtype=x.dtype) # [B x curr x 1 x n_heads]
        return (
            jnp.concatenate([zero_pad, x], axis=2).reshape((x.shape[2] + 1, *x.shape[:2], x.shape[-1]))[1:].reshape(x.shape)
        )

    def __call__(
        self, 
        input_: float, 
        pos_embs: float, 
        memory: float, 
        u: float, 
        v: float, 
        mask: float = None
    ):
        """
        + pos_embs: positional embeddings passed separately to handle relative positions.
        + Arguments
            - input:  (bs, seq, self.d_input)
            - pos_embs: (bs, seq + prev_seq, self.d_input)
            - memory: (bs, prev_seq, self.d_input)
            - u: (num_heads, inner_dim)
            - v: (num_heads, inner_dim)
            - mask: (seq, seq + prev_seq, 1)
        + Returns
            - output: (bs, seq, self.d_input)
        + symbols representing shape of the tensors
            - cs: current sequence length, b: batch, H: no. of heads
            - d: inner dimension, ps: previous sequence length
        """
        cur_seq = input_.shape[1]
        prev_seq = memory.shape[1]
        H, d = self.n_heads, self.d_head_inner
        # concat memory across sequence dimension
        # input_with_memory =  [B, seq + prev_seq x d_input]
        input_with_memory = jnp.concatenate([memory, input_], axis=1)

        # k_tfmd, v_tfmd = [B x seq + prev_seq x n_heads.d_head_inner]
        k_tfmd, v_tfmd = jnp.split(
            self.linear_kv(input_with_memory),
            2,
            axis=-1,
        )
        # q_tfmd = [B x seq x n_heads.d_head_inner]
        q_tfmd = self.linear_q(input_)

        bs, _, _ = q_tfmd.shape
        assert bs == k_tfmd.shape[0]

        # content_attn = [B x curr x curr+prev x n_heads]
        content_attn = jnp.einsum(
            "bihd,bjhd->bijh",
                q_tfmd.reshape((bs, cur_seq, H, d)) + u,
                k_tfmd.reshape((bs, cur_seq + prev_seq, H, d)),
        )

        # p_tfmd: [1 x seq + prev_seq x n_heads.d_head_inner]
        p_tfmd = self.linear_p(pos_embs)
        # position_attn = [B x curr x curr+prev x n_heads]
        position_attn = jnp.einsum(
            "bihd,jhd->bijh",
                q_tfmd.reshape((bs, cur_seq, H, d)) + v,
                p_tfmd.reshape((cur_seq + prev_seq, H, d)),
        )

        position_attn = self._rel_shift(position_attn)
        # attn = [B x curr x curr+prev x n_heads]
        attn = content_attn + position_attn
        assert attn.shape == (bs, cur_seq, cur_seq + prev_seq, H)

        if mask is not None and mask.any():
            # fills float('-inf') where mask is True.
            attn = jax.lax.select(
                jax.lax.broadcast_in_dim(mask, attn.shape, (1,2,3)), 
                attn, 
                jax.lax.broadcast(-1e30, attn.shape)
            )
        # rescale to prevent values from exploding.
        # normalize across the value sequence dimension.
        attn = jax.nn.softmax(attn * self.scale, axis=2)
        # attn = [B x curr x curr+prev x n_heads]
        attn = hk.dropout(rng=hk.next_rng_key(), rate=self.dropa, x=attn)

        # attn_weighted_values = [B x curr x n_heads.d_inner]
        attn_weighted_values = (
            jnp.einsum(
                "bijh,bjhd->bihd",
                    attn,  # (b, cs, cs + ps, H)
                    v_tfmd.reshape((bs, cur_seq + prev_seq, H, d)),  # (b, cs + ps, H, d)
                # (b, cs, H, d)
            ).reshape(bs, cur_seq, H * d)
        )  # (b, cs, H * d)

        # output = [B x curr x d_input]
        output = self.lout(attn_weighted_values)
        output = hk.dropout(rng=hk.next_rng_key(), rate=self.dropo, x=output)
        return output

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
        dropouta: float = 0.0
    ):
        super().__init__()
        self.layernorm1 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True) 
        self.layernorm2 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        self.mha = MultiHeadAttentionXL(
            d_input=model_dimension,
            d_head_inner=key_size,
            n_heads=num_heads,
            dropout=dropout,
            dropouta=dropouta,
        )
        self.gated_mha = GRUGate(input_dimension=model_dimension)
        self.mlp = hk.nets.MLP(hidden_sizes_mlp+[model_dimension])
        self.gated_mlp = GRUGate(input_dimension=model_dimension)


    def __call__(
        self,
        e_prev,
        pos_embs, 
        u, 
        v, 
        mask=None, 
        mems=None
    ):
        e_tilde_prev_norm = self.layernorm1(e_prev)
        y_bar = self.mha(e_tilde_prev_norm, pos_embs, mems, u, v, mask=mask)
        y = self.gated_mha(e_prev, jax.nn.relu(y_bar))
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
        hidden_sizes_mlp: Iterable[int] = [],
        dropouta: float = 0.0,
    ):
        self.model_dimension = model_dimension
        self.num_heads = num_heads 
        self.key_size = key_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.hidden_sizes_mlp = hidden_sizes_mlp
        self.dropouta = dropouta
        self.positional_encoding = PositionalEncoding(dim=model_dimension)

        init = hk.initializers.TruncatedNormal(stddev=0.1, mean=0.)
        self.u = hk.get_parameter(f"u", [num_heads, key_size], jnp.float32, init=init)
        self.v = hk.get_parameter(f"v", [num_heads, key_size], jnp.float32, init=init)

        self.layers = [
                GTrXLCore(
                    model_dimension=self.model_dimension, 
                    num_heads=self.num_heads, 
                    key_size=self.key_size, 
                    dropout=self.dropout, 
                    hidden_sizes_mlp=self.hidden_sizes_mlp,
                    dropouta = self.dropouta
                )
                for _ in range(num_layers)
            ]
        

    def init_memory(self, x):
        return [
            jnp.zeros(x.shape, dtype=jnp.float32)
            for _ in range(self.num_layers + 1)
        ]

    def update_memory(
        self, 
        previous_memory: Tuple[jnp.ndarray], 
        hidden_states: Tuple[jnp.ndarray]
    ):
        assert len(hidden_states) == len(previous_memory)
        mem_len, seq_len = previous_memory[0].shape[1], hidden_states[0].shape[1]

        new_memory = []
        end_idx = mem_len + seq_len
        beg_idx = max(0, end_idx - mem_len)
        for m, h in zip(previous_memory, hidden_states):
            cat = jnp.concatenate([m, h], axis=1)
            new_memory.append(cat[:, beg_idx:end_idx])
        return jax.lax.stop_gradient(new_memory)

    def forward(
        self, 
        inputs: jnp.ndarray, 
        memory: Tuple[jnp.ndarray] = None,
    ):
        """
        + Arguments
            - inputs: [B, T, d_inner]
            - memory: [[B, T, d_inner] x num_layers + 1]
        """
        if memory is None:
            memory = self.init_memory(inputs)
        assert len(memory) == len(self.layers) + 1

        bs, cur_seq = inputs.shape[:2]
        prev_seq = memory[0].shape[1]

        # dec_attn_mask = [curr x curr + prev x 1] 
        # if jnp instead of np here -> ConcretizationTypeError when jitted
        dec_attn_mask = (
            np.triu(
                np.ones((cur_seq, cur_seq + prev_seq)),
                k=1 + prev_seq,
            ).astype(bool)[..., None]
        )
        #dec_attn_mask = None # still to understand if mask makes any difference

        pos_ips = jnp.arange(cur_seq + prev_seq - 1, -1, -1.0, dtype=jnp.float32)
        # pos_embs = [curr + prev x 1 x d_input]
        pos_embs = hk.dropout(rng=hk.next_rng_key(), rate=self.dropout, x=self.positional_encoding(pos_ips))
        if self.model_dimension % 2 != 0:
            pos_embs = pos_embs[:, :, :-1]

        hidden_states = [inputs]
        layer_out = inputs
        for mem, layer in zip(memory, self.layers):
            # layer_out = [B x curr x d_inner]
            layer_out = layer(
                layer_out,
                pos_embs,
                self.u,
                self.v,
                mask=dec_attn_mask,
                mems=mem,
            )
            hidden_states.append(layer_out)

        # Memory is treated as a const., don't propagate through it
        # new_memory = [[B x T x d_inner] x num_layers+1]
        memory = self.update_memory(memory, hidden_states)
        return {"outs": layer_out, "memory": memory}


GTrXLApply = namedtuple(
    'GTrXLApply',
    ['forward']
)

def apply_GTrXL(model_dimension, num_heads, key_size, num_layers, dropout, hidden_sizes_mlp):
  transformer = GTrXL(model_dimension, num_heads, key_size, num_layers, dropout, hidden_sizes_mlp)
  def init(x):
    return transformer.forward(x)

  return init, GTrXLApply(transformer.forward)
