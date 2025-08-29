# jax_llm_posttrain.py
"""
JAX-native LLM Post-Training Utilities

Provides:
- A tiny LoRA-style adapter helper (add, apply, merge)
- Simple post-training symmetric per-tensor quantization (int8)
- Small prune helper

Notes:
- This is intentionally straightforward and dependency-light (JAX + numpy).
- Does not touch optimizer state; these are parameter transforms.
- For real large models you may want TensorStore-backed IO, sharding, distributed merging, etc.

Dependencies:
- jax
- numpy
"""

from typing import Dict, Tuple, Any
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np


# ---------- LoRA-style adapters ----------
@dataclasses.dataclass
class LoRAConfig:
    r: int = 4               # adapter rank
    alpha: float = 1.0       # scaling factor
    dtype: Any = jnp.float32


def init_lora_for_param(param: jnp.ndarray, cfg: LoRAConfig, key: jax.random.KeyArray) -> Dict[str, jnp.ndarray]:
    """
    Initialize LoRA adapters for a single weight matrix `param` of shape (in, out).
    Returns dict with 'A' and 'B' (down and up matrices) and scale.
    """
    in_dim, out_dim = param.shape
    k1, k2 = jax.random.split(key)
    A = jax.random.normal(k1, (in_dim, cfg.r), dtype=cfg.dtype) * (1.0 / jnp.sqrt(in_dim))
    B = jax.random.normal(k2, (cfg.r, out_dim), dtype=cfg.dtype) * (1.0 / jnp.sqrt(cfg.r))
    scale = cfg.alpha / cfg.r
    return {"A": A, "B": B, "scale": jnp.array(scale, dtype=cfg.dtype)}


def apply_lora_to_param(param: jnp.ndarray, lora: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    """
    Apply LoRA update (non-merged) to a weight matrix:
      W' = W + scale * (A @ B)
    """
    delta = jnp.dot(lora["A"], lora["B"]) * lora["scale"]
    return param + delta


def add_lora(params: Dict[str, Any], cfg: LoRAConfig, rng: jax.random.KeyArray) -> Dict[str, Any]:
    """
    Walk model params dict and add LoRA entries for 2D weight matrices.
    Returns a new dict mapping same keys to either None or lora dict.
    Example shape assumption: params is nested mapping -> leaf arrays.
    """
    flat, treedef = jax.tree_flatten(params)
    keys = list(range(len(flat)))
    rngs = jax.random.split(rng, len(flat))
    lora_leaves = []
    for i, leaf in enumerate(flat):
        if leaf.ndim == 2:
            lora_leaves.append(init_lora_for_param(leaf, cfg, rngs[i]))
        else:
            lora_leaves.append(None)
    return jax.tree_unflatten(treedef, lora_leaves)


def merge_lora_into_params(params: Dict[str, Any], lora_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge LoRA adapters into the base params, returning merged params.
    """
    def merge_fn(p, l):
        return apply_lora_to_param(p, l) if (l is not None) else p
    return jax.tree_map(merge_fn, params, lora_state)


# ---------- Simple symmetric per-tensor quantization ----------
@dataclasses.dataclass
class QuantizedTensor:
    scale: jnp.ndarray
    zero_point: jnp.ndarray  # unused for symmetric quantization but left for extensibility
    qtensor: jnp.ndarray     # int8 tensor stored as int8 view


def quantize_tensor_symmetric(x: jnp.ndarray, num_bits: int = 8) -> QuantizedTensor:
    """
    Symmetric per-tensor quantization to signed integers (int8 when num_bits=8).
    Stores scale and integer tensor. This is simple and not optimal for all cases.
    """
    qmax = 2 ** (num_bits - 1) - 1
    absmax = jnp.max(jnp.abs(x))
    scale = absmax / qmax if absmax > 0 else jnp.array(1.0, dtype=x.dtype)
    q = jnp.round(x / scale).astype(jnp.int8)
    # zero_point kept for API parity; 0 for symmetric
    zero_point = jnp.array(0, dtype=jnp.int32)
    return QuantizedTensor(scale=scale.astype(jnp.float32), zero_point=zero_point, qtensor=q)


def dequantize_tensor(q: QuantizedTensor) -> jnp.ndarray:
    return q.qtensor.astype(jnp.float32) * q.scale


def quantize_params(params: Dict[str, Any], num_bits: int = 8) -> Dict[str, Any]:
    """
    Quantize all ndarray leaves in params to QuantizedTensor wrappers.
    Leaves that are not float arrays are left untouched.
    """
    def q_fn(p):
        if isinstance(p, jnp.ndarray) or isinstance(p, np.ndarray):
            if jnp.issubdtype(p.dtype, jnp.floating):
                return quantize_tensor_symmetric(jnp.array(p), num_bits=num_bits)
        return p
    return jax.tree_map(q_fn, params)


def dequantize_params(qparams: Dict[str, Any]) -> Dict[str, Any]:
    def dq_fn(p):
        if isinstance(p, QuantizedTensor):
            return dequantize_tensor(p)
        return p
    return jax.tree_map(dq_fn, qparams)


# ---------- Tiny pruning helper ----------
def magnitude_prune(params: Dict[str, Any], fraction: float = 0.1) -> Dict[str, Any]:
    """
    Zero-out the smallest `fraction` of absolute weights in each weight matrix leaf.
    Returns pruned params.
    """
    def prune_leaf(p):
        if isinstance(p, jnp.ndarray) and jnp.issubdtype(p.dtype, jnp.floating):
            flat = jnp.abs(p).ravel()
            k = max(1, int(jnp.ceil(flat.size * fraction)))
            if k >= flat.size:
                return jnp.zeros_like(p)
            thresh = jnp.sort(flat)[k - 1]
            mask = jnp.where(jnp.abs(p) > thresh, 1.0, 0.0)
            return p * mask
        return p
    return jax.tree_map(prune_leaf, params)


# ---------- Example usage ----------
if __name__ == "__main__":
    # minimal example demonstrating LoRA add/apply and quantize/dequantize
    key = jax.random.PRNGKey(42)

    # Fake "model params": nested dict with a 2D weight and a bias vector
    params = {
        "layer1": {"w": jnp.array(jax.random.normal(key, (16, 32))), "b": jnp.zeros((32,))},
        "layer2": {"w": jnp.array(jax.random.normal(key, (32, 64))), "b": jnp.zeros((64,))}
    }
    cfg = LoRAConfig(r=8, alpha=8.0)

    # init lora adapters
    lora_state = add_lora(params, cfg, key)

    # apply lora (non-destructively) to obtain adapted params
    adapted = merge_lora_into_params(params, lora_state)
    # check shapes
    print("Original layer1.w shape:", params["layer1"]["w"].shape)
    print("Adapted layer1.w shape:", adapted["layer1"]["w"].shape)

    # quantize adapted params
    qparams = quantize_params(adapted, num_bits=8)
    # dequantize back
    deq = dequantize_params(qparams)
    # print a small sanity check
    diff = jnp.max(jnp.abs(deq["layer1"]["w"] - adapted["layer1"]["w"]))
    print("Max reconstruction error after quantize/dequantize (layer1.w):", float(diff))

    # pruning example (10% smallest weights zeroed)
    pruned = magnitude_prune(adapted, fraction=0.1)
    print("Pruned layer1.w nonzeros:", jnp.sum(pruned["layer1"]["w"] != 0))
