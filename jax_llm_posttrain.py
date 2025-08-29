# jax_llm_posttrain.py
"""
JAX-native LLM Post-Training Utilities

Key improvements:
- safer symmetric per-tensor quantization (clipped integer range, explicit dtype handling)
- LoRA init has assertion and optional transpose handling
- pruning and quant functions use jax.numpy ops consistently
- clearer docstrings and type hints
"""

from typing import Dict, Any, Optional
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np


@dataclasses.dataclass
class LoRAConfig:
    r: int = 4
    alpha: float = 1.0
    dtype: Any = jnp.float32
    assume_shape: str = "in_out"  # "in_out" assumes param shape (in_dim, out_dim); "out_in" handles (out,in)


def init_lora_for_param(param: jnp.ndarray, cfg: LoRAConfig, key: jax.random.KeyArray) -> Dict[str, jnp.ndarray]:
    """
    Initialize LoRA A,B for a 2D weight matrix.
    By default we expect param shape (in_dim, out_dim). If your weights are (out, in), set cfg.assume_shape="out_in".
    """
    assert param.ndim == 2, "LoRA adapter only for 2D weight matrices"
    if cfg.assume_shape == "in_out":
        in_dim, out_dim = param.shape
    else:
        out_dim, in_dim = param.shape
    k1, k2 = jax.random.split(key)
    A = jax.random.normal(k1, (in_dim, cfg.r), dtype=cfg.dtype) * (1.0 / jnp.sqrt(max(1, in_dim)))
    B = jax.random.normal(k2, (cfg.r, out_dim), dtype=cfg.dtype) * (1.0 / jnp.sqrt(max(1, cfg.r)))
    scale = jnp.array(cfg.alpha / max(1, cfg.r), dtype=cfg.dtype)
    return {"A": A, "B": B, "scale": scale, "assume_shape": cfg.assume_shape}


def apply_lora_to_param(param: jnp.ndarray, lora: Optional[Dict[str, jnp.ndarray]]) -> jnp.ndarray:
    """
    Apply LoRA delta. If lora is None, return original param.
    """
    if lora is None:
        return param
    assume_shape = lora.get("assume_shape", "in_out")
    A = lora["A"]
    B = lora["B"]
    delta = jnp.dot(A, B) * lora["scale"]
    if assume_shape == "out_in":
        # delta is (in, out) computed from assumed transpose; transpose to match param shape (out, in)
        delta = delta.T
    return param + delta


def add_lora(params: Dict[str, Any], cfg: LoRAConfig, rng: jax.random.KeyArray) -> Dict[str, Any]:
    flat, treedef = jax.tree_flatten(params)
    rngs = jax.random.split(rng, len(flat))
    lora_leaves = []
    for i, leaf in enumerate(flat):
        if isinstance(leaf, (jnp.ndarray, np.ndarray)) and getattr(leaf, "ndim", None) == 2:
            lora_leaves.append(init_lora_for_param(jnp.array(leaf), cfg, rngs[i]))
        else:
            lora_leaves.append(None)
    return jax.tree_unflatten(treedef, lora_leaves)


def merge_lora_into_params(params: Dict[str, Any], lora_state: Dict[str, Any]) -> Dict[str, Any]:
    def merge_fn(p, l):
        return apply_lora_to_param(p, l)
    return jax.tree_map(merge_fn, params, lora_state)


@dataclasses.dataclass
class QuantizedTensor:
    scale: jnp.ndarray
    zero_point: jnp.ndarray
    qtensor: jnp.ndarray  # int8 view


def quantize_tensor_symmetric(x: jnp.ndarray, num_bits: int = 8) -> QuantizedTensor:
    """
    Symmetric per-tensor quantization to signed integers.
    Returns QuantizedTensor where qtensor dtype is jnp.int8 (when num_bits==8).
    """
    assert x.dtype in (jnp.float32, jnp.float64, jnp.bfloat16) or jnp.issubdtype(x.dtype, jnp.floating)
    qmax = 2 ** (num_bits - 1) - 1
    absmax = jnp.max(jnp.abs(x))
    # Avoid divide by zero
    scale = jnp.where(absmax > 0, absmax / qmax, jnp.array(1.0, dtype=x.dtype))
    scaled = jnp.round(x / scale)
    clipped = jnp.clip(scaled, -qmax - 1, qmax)  # keep in signed range; int8 range is -128..127
    qt = clipped.astype(jnp.int8)
    zero_point = jnp.array(0, dtype=jnp.int32)
    return QuantizedTensor(scale=scale.astype(jnp.float32), zero_point=zero_point, qtensor=qt)


def dequantize_tensor(q: QuantizedTensor) -> jnp.ndarray:
    return q.qtensor.astype(jnp.float32) * q.scale


def quantize_params(params: Dict[str, Any], num_bits: int = 8) -> Dict[str, Any]:
    def q_fn(p):
        if isinstance(p, (jnp.ndarray, np.ndarray)) and jnp.issubdtype(jnp.array(p).dtype, jnp.floating):
            return quantize_tensor_symmetric(jnp.array(p), num_bits=num_bits)
        return p
    return jax.tree_map(q_fn, params)


def dequantize_params(qparams: Dict[str, Any]) -> Dict[str, Any]:
    def dq_fn(p):
        if isinstance(p, QuantizedTensor):
            return dequantize_tensor(p)
        return p
    return jax.tree_map(dq_fn, qparams)


def magnitude_prune(params: Dict[str, Any], fraction: float = 0.1) -> Dict[str, Any]:
    """Zero-out smallest `fraction` absolute weights per-array leaf."""
    def prune_leaf(p):
        if isinstance(p, (jnp.ndarray, np.ndarray)) and jnp.issubdtype(jnp.array(p).dtype, jnp.floating):
            arr = jnp.array(p)
            flat = jnp.abs(arr).ravel()
            k = int(jnp.ceil(flat.size * fraction))
            if k <= 0:
                return arr
            if k >= flat.size:
                return jnp.zeros_like(arr)
            thresh = jnp.sort(flat)[k - 1]
            mask = jnp.where(jnp.abs(arr) > thresh, 1.0, 0.0)
            return arr * mask
        return p
    return jax.tree_map(prune_leaf, params)


if __name__ == "__main__":
    key = jax.random.PRNGKey(42)
    # Fake "model params": nested dict with 2D weights
    params = {
        "layer1": {"w": jnp.array(jax.random.normal(key, (16, 32))), "b": jnp.zeros((32,))},
        "layer2": {"w": jnp.array(jax.random.normal(key, (32, 64))), "b": jnp.zeros((64,))}
    }
    cfg = LoRAConfig(r=8, alpha=8.0)

    lora_state = add_lora(params, cfg, key)
    adapted = merge_lora_into_params(params, lora_state)

    qparams = quantize_params(adapted, num_bits=8)
    deq = dequantize_params(qparams)
    diff = jnp.max(jnp.abs(deq["layer1"]["w"] - adapted["layer1"]["w"]))
    print("Max reconstruction error after quantize/dequantize (layer1.w):", float(diff))

    pruned = magnitude_prune(adapted, fraction=0.1)
    print("Pruned layer1.w nonzeros:", int(jnp.sum(pruned["layer1"]["w"] != 0)))
