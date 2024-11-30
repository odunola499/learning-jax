from flax import nnx
import jax
from jax import numpy as jnp
from jax import random
import math
from dataclasses import dataclass
from functools import partial
import optax


@dataclass
class GPTConfig:
    block_size: int = 64
    vocab_size: int = 500
    n_layer: int = 2
    n_head: int = 2
    n_embd: int = 32
    dropout: float = 0.0
    attn_dropout: float = 0.0
    resid_dropout: float = 0.0
    bias: bool = True


def _init_weights(module, key):
    if isinstance(module, nnx.Linear):
        module.kernel.value = random.normal(key, shape = module.kernel.shape) * 0.02
        if module.bias.value is not None:
            module.bias.value = jnp.zeros_like(module.bias)
    elif isinstance(module, nnx.Embed):
        module.kernel.value = random.normal(key, shape = module.kernel.shape) * 0.02
    elif isinstance(module, LayerNorm):
        module.kernel.value = jnp.ones_like(module.kernel, device = module.kernel.device)
        if module.bias.value is not None:
            module.bias = jnp.zeros_like(module.bias)


class AttrDict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"No such attribute: {name}")

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError:
            raise AttributeError(f"No such attribute: {name}")

    @classmethod
    def from_dict(cls, obj):
        if isinstance(obj, dict):
            return cls({k: cls.from_dict(v) for k, v in obj.items()})
        elif isinstance(obj, list):
            return [cls.from_dict(v) if isinstance(v, dict) else v for v in obj]
        return obj


class LayerNorm(nnx.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nnx.Param(jnp.ones(ndim))
        self.bias = nnx.Param(jnp.zeros(ndim)) if bias else None

    def __call__(self, x, epsilon = 1e-5):
        mean = jnp.mean(x, axis=-1, keepdims = True)
        variance = jnp.var(x, axis=-1, keepdims = True)
        normalized = (x - mean) / jnp.sqrt(variance + epsilon)
        output = normalized * self.weight + (self.bias if self.bias else 0)
        return output

class CausalSelfAttention(nnx.Module):
    def __init__(self, config, rngs:dict = None):
        super().__init__()
        if rngs is None:
            rngs = nnx.Rngs(0)

        assert config.n_embd % config.n_head == 0
        self.c_attn = nnx.Linear(config.n_embd, 3 * config.n_embd, use_bias = config.bias,
                                 rngs = rngs)
        self.c_proj = nnx.Linear(config.n_embd, config.n_embd, use_bias = config.bias,
                                 rngs = rngs)
        self.attn_dropout = nnx.Dropout(config.attn_dropout)
        self.resid_dropout = nnx.Dropout(config.resid_dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.bias = jnp.tril(jnp.ones((config.block_size, config.block_size))).reshape(1, 1, config.block_size, config.block_size)


    def __call__(self,x:jnp.array, training = True):
        B, T, C = x.shape
        c = self.c_attn(x)
        q,k,v = jnp.split(c, 3, axis = 2)
        k = jnp.reshape(k, (B, T, self.n_head, C // self.n_head))
        q = jnp.reshape(q, (B, T, self.n_head, C // self.n_head))
        v = jnp.reshape(v, (B, T, self.n_head, C // self.n_head))

        k = jnp.transpose(k, (0,2,1,3))
        q = jnp.transpose(q, (0,2,1,3))
        v = jnp.transpose(v, (0,2,1,3))

        attn = jnp.einsum('bhqd, bhkd -> bhqk', q,k) * (1.0 /math.sqrt(C))
        causal_mask = self.bias[:,:,:T, :T]
        attn = jnp.where(causal_mask == 0, -jnp.inf, attn)
        attn = jax.nn.softmax(attn, axis = -1)
        if training:
            attn = self.attn_dropout(attn)
        y = jnp.einsum('bhqk, bhvd -> bhqd', attn, v)
        y = y.transpose(0,2,1,3).reshape(B,T,C)
        y = self.c_proj(y)
        if training:
            y = self.resid_dropout(y)
        return y


class MLP(nnx.Module):
    def __init__(self, config, rngs = None):
        super().__init__()
        if rngs is None:
            rngs = nnx.Rngs(0)
        self.c_fc = nnx.Linear(config.n_embd, 4 * config.n_embd, use_bias = config.bias,
                               rngs = rngs)
        self.gelu = partial(nnx.gelu)
        self.c_proj = nnx.Linear(4 * config.n_embd, config.n_embd, use_bias = config.bias,
                                 rngs = rngs)
        self.dropout = nnx.Dropout(config.dropout)

    def __call__(self,x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nnx.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.ln_1 = LayerNorm(config.n_embd, bias = config.bias)
        self.ln_2 = LayerNorm(config.n_embd, bias = config.bias)

    def __call__(self, x):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nnx.Module):
    def __init__(self, config, rngs = None, init_weights=None):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        if rngs is None:
            rngs = nnx.Rngs(0)
        self.rngs = rngs


        self.transformer = dict(
            wte = nnx.Embed(config.vocab_size, config.n_embd, rngs = self.rngs),
            wpe = nnx.Embed(config.block_size, config.n_embd, rngs = self.rngs),
            drop = nnx.Dropout(config.dropout),
            h = [Block(config) for _ in range(config.n_layer)],
            ln_f = LayerNorm(config.n_embd, bias = config.bias)
        )
        self.transformer = AttrDict.from_dict(self.transformer)

        self.lm_head = nnx.Linear(config.n_embd, config.vocab_size, use_bias = False, rngs = self.rngs)

        if init_weights:
            self.apply_init(init_weights)

    def apply_init(self, init_fn):
        for module_name, module in self.__dict__.items():
            if isinstance(module, nnx.Module):
                init_fn(module)

    def __call__(self, idx, target = None):
        device = idx.device
        b,t = idx.shape
        assert t <= self.config.block_size
        pos = jnp.arange(0, t, dtype = jnp.int64, device = device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        if target is not None:
            logits = self.lm_head(x)
            loss = optax.softmax_cross_entropy_with_integer_labels(
                jnp.reshape(logits, (-1, logits.shape[-1])),
                jnp.reshape(target, (-1,))
            ).mean()
        else:
            logits = self.lm_head(x[:,[-1], :])
            loss = None

        return logits, loss




if __name__ == "__main__":
    config = GPTConfig()
    rng = nnx.Rngs(0)
    init_rng = random.PRNGKey(0)
    model = GPT(config, rngs=rng, init_weights=lambda mod: _init_weights(mod,init_rng))
    print("Model initialized successfully.")
    x = random.randint(init_rng, (2,5), minval = 0, maxval = 10, dtype = jnp.int64)
    y = random.randint(init_rng, (2,5), minval = 0, maxval = 10, dtype = jnp.int64)
    logits, loss = model(x, target = y)
    print(logits)
    print(loss)



