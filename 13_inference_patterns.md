---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Part 13: Capstone - LLM Inference

> **Capstone Project**: The first 12 notebooks taught the core primitives of JAX: arrays, compilation, buffer donation, sharding, profiling, and custom kernels. Now we put it all together. This notebook is a fully working, non-trivial application of those concepts to a modern workload: autoregressive LLM inference.

> **Hardware note**: These patterns generalize across CPU, GPU, and TPU, but several design choices (bfloat16 native support, shape-aware compilation, large memory hierarchies, fast accelerator buffers) have the biggest impact on accelerators. On CPU, the timings shown are illustrative: they demonstrate the *technique*, not the achievable speedup.

```{code-cell} ipython3
# @title Setup { display-mode: "form" }

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import time

print(f"JAX version:     {jax.__version__}")
print(f"Devices:         {jax.devices()}")
print(f"Default backend: {jax.default_backend()}")
```

---

# 1. The Scaffold: A Single Decoder Block

Throughout this notebook we'll work with a minimal **single-layer decoder block**: the kind of building block that appears in modern transformer architectures. The architecture is:

```
Input x (shape: seq_len × d_model)
    │
    ▼
RMSNorm
    │
    ▼
Causal single-head self-attention
    │   │
    │   └── Output projection
    ▼
   (+) residual
    │
    ▼
RMSNorm
    │
    ▼
MLP: Dense → SiLU → Dense
    │
    ▼
   (+) residual
    │
    ▼
Output (shape: seq_len × d_model)
```

This is **not** a full transformer: there's no positional encoding, no multi-head attention, no embedding layer. The point is to have something realistic enough that the optimizations in the rest of the notebook have a believable target.

**Style choices:**
- High-level structure uses Flax (`nn.Module`, `nn.Dense`, `init`/`apply`) for consistency with Notebooks 08-11.

- Attention math is written in raw JAX rather than using `nn.MultiHeadDotProductAttention`, so the internals are inspectable.

- Sizes are chosen so the notebook runs in seconds on CPU: `d_model = 128`, `mlp_hidden = 256`, `seq_len = 64`.

> **What's RMSNorm?** Root-mean-square layer norm: scales each row of the input by its inverse RMS. It's a simpler, popular alternative to LayerNorm in modern transformers. The exact form doesn't matter for the optimizations here; treat it as a stand-in for "some normalization step."

> **What's SiLU?** `x * sigmoid(x)`. A smooth alternative to ReLU. Again, the choice doesn't matter for our purposes; treat it as "some non-linear activation."

```{code-cell} ipython3
# @title Decoder Block Definition { display-mode: "form" }

D_MODEL = 128
MLP_HIDDEN = 256
SEQ_LEN = 64


def rms_norm(x, eps=1e-6):
    """Root-mean-square layer norm (raw JAX)."""
    variance = jnp.mean(x ** 2, axis=-1, keepdims=True)
    return x * jax.lax.rsqrt(variance + eps)


def causal_attention(q, k, v):
    """Single-head causal self-attention (raw JAX).

    q, k, v: shape (seq_len, d_model).
    Returns: shape (seq_len, d_model).
    """
    d = q.shape[-1]
    scores = (q @ k.T) / jnp.sqrt(d)                        # (seq_len, seq_len)
    mask = jnp.tril(jnp.ones(scores.shape, dtype=bool))     # position i sees j ≤ i
    scores = jnp.where(mask, scores, -jnp.inf)
    attn = jax.nn.softmax(scores, axis=-1)
    return attn @ v                                          # (seq_len, d_model)


class DecoderBlock(nn.Module):
    d_model: int = D_MODEL
    mlp_hidden: int = MLP_HIDDEN

    @nn.compact
    def __call__(self, x):
        # Attention sub-layer (pre-norm + residual)
        h = rms_norm(x)
        q = nn.Dense(self.d_model, name='q_proj', use_bias=False)(h)
        k = nn.Dense(self.d_model, name='k_proj', use_bias=False)(h)
        v = nn.Dense(self.d_model, name='v_proj', use_bias=False)(h)
        attn_out = causal_attention(q, k, v)
        x = x + nn.Dense(self.d_model, name='o_proj', use_bias=False)(attn_out)

        # MLP sub-layer (pre-norm + residual)
        h = rms_norm(x)
        h = nn.Dense(self.mlp_hidden, name='mlp_in')(h)
        h = jax.nn.silu(h)
        h = nn.Dense(self.d_model, name='mlp_out')(h)
        return x + h


# Initialize the block
model = DecoderBlock()
key = jax.random.key(0)
x_dummy = jnp.ones((SEQ_LEN, D_MODEL))
params = model.init(key, x_dummy)['params']

n_params = sum(a.size for a in jax.tree.leaves(params))
print(f"Decoder block initialized:")
print(f"  d_model={D_MODEL}, mlp_hidden={MLP_HIDDEN}, seq_len={SEQ_LEN}")
print(f"  Total parameters: {n_params:,}")
print(f"\nParameter shapes:")
for layer_name, layer_params in params.items():
    for param_name, arr in layer_params.items():
        print(f"  {layer_name}/{param_name}: {tuple(arr.shape)}")
```

---

# 2. Applying Precision Strategies

In Notebook 11, we saw how **bfloat16** and **int8** quantization drastically cut memory usage. Because inference is usually memory-bandwidth bound, applying precision strategies is the first step to deploying a model.

Let's apply the standard bfloat16 inference recipe to our DecoderBlock: **cast parameters to bf16 and leave the input dtype unchanged**.

```{code-cell} ipython3
# Cast every parameter leaf to bf16. The model definition is unchanged -
# only the dtype of the params pytree changes.
params_bf16 = jax.tree.map(lambda a: a.astype(jnp.bfloat16), params)

f32_bytes  = sum(a.nbytes for a in jax.tree.leaves(params))
bf16_bytes = sum(a.nbytes for a in jax.tree.leaves(params_bf16))

print(f"float32  params: {f32_bytes / 1024:.1f} KB")
print(f"bfloat16 params: {bf16_bytes / 1024:.1f} KB  ({bf16_bytes / f32_bytes:.0%} of float32)")

# Forward passes match closely. (The bf16 version returns bf16; cast back for comparison.)
x = jax.random.normal(jax.random.key(1), (SEQ_LEN, D_MODEL))
out_f32  = model.apply({'params': params}, x)
out_bf16 = model.apply({'params': params_bf16}, x.astype(jnp.bfloat16)).astype(jnp.float32)

rel_err = jnp.mean(jnp.abs(out_f32 - out_bf16) / (jnp.abs(out_f32) + 1e-6))
print(f"\nRelative error (bf16 vs f32 output): {rel_err.item():.4f}")
print("Typical bf16 inference loss is well below what affects downstream metrics.")
```

---

# 3. Serving with Bucketing

Recall from Notebook 11: **every new input shape triggers a fresh trace and compile**. For training that's usually fine: shapes are stable batch to batch. For inference it's a problem, because real requests come in with variable shapes (sequence lengths).

We apply **bucketing**: define a small set of allowed sequence lengths, pad each incoming request up to the nearest bucket, and run the same compiled artifact.

```
                              compile once per bucket
                              ┌────────────────────┐
  request: seq_len = 47   ──► │ artifact[seq=64]   │
  request: seq_len = 22   ──► │ artifact[seq=32]   │
  request: seq_len = 5    ──► │ artifact[seq=16]   │
  request: seq_len = 200  ──► │ artifact[seq=256]  │
                              └────────────────────┘
                         padded up to nearest bucket
```

```{code-cell} ipython3
BUCKETS = (16, 32, 64, 128, 256)


def pick_bucket(actual_len, buckets=BUCKETS):
    """Return the smallest bucket size >= actual_len. Raises if none fits."""
    for b in buckets:
        if b >= actual_len:
            return b
    raise ValueError(f"No bucket >= {actual_len}; largest is {buckets[-1]}")


def pad_to_bucket(x, target_len):
    """Pad a (seq_len, d_model) tensor along axis 0 up to target_len.

    Returns (padded_x, mask) where mask is (target_len,) with True for real positions.
    """
    seq_len = x.shape[0]
    pad_amount = target_len - seq_len
    padded = jnp.pad(x, ((0, pad_amount), (0, 0)))
    mask = jnp.arange(target_len) < seq_len
    return padded, mask


# One compiled artifact per bucket size
compiled_artifacts = {
    b: jax.jit(lambda p, x, _model=model: _model.apply({'params': p}, x))
    for b in BUCKETS
}

# Simulate a small stream of variable-length requests
rng = np.random.default_rng(0)
request_lengths = [rng.integers(1, 250) for _ in range(8)]

print(f"{'request len':>12}  {'bucket':>8}  {'padded shape':>15}  {'masked positions':>20}")
print("-" * 64)
for actual_len in request_lengths:
    bucket = pick_bucket(actual_len)
    x_req = jax.random.normal(jax.random.key(int(actual_len)), (actual_len, D_MODEL))
    x_padded, mask = pad_to_bucket(x_req, bucket)
    y = compiled_artifacts[bucket](params, x_padded)
    # In a real server, you'd drop the padded positions from y using mask before returning.
    print(f"{actual_len:>12}  {bucket:>8}  {str(x_padded.shape):>15}  "
          f"{int(mask.sum()):>10} / {bucket}")

print(f"\nCompiled artifacts: {len(BUCKETS)}, one per bucket.")
print("Each request reuses the artifact for its bucket - no per-request compilation.")
```

> **Caveat on padded positions**: The decoder block's `causal_attention` only applies a causal mask, not a padding mask. In a real serving setup you'd extend the mask to also zero out attention to padded positions, so the model output for real positions isn't influenced by garbage padding. We've left that out to keep the decoder block in Section 1 small; the bucketing pattern itself is unchanged.

+++

---

# 4. KV-Cache with Buffer Donation

## 4.1 Why a KV-Cache?

Autoregressive generation means "produce one token, then use what you've produced so far as input to produce the next one, then repeat". Inside the attention math, every token contributes a **K** (key) vector and a **V** (value) vector. 

The naïve setup: at step `t`, you re-run the whole model on tokens `0..t-1`. The work scales as O(t²).
The fix: cache the keys and values you've already computed. At each step you compute K and V *only for the new token*, write them into a pre-allocated buffer, then attend over the filled portion.

```
Step 0:  compute K[0], V[0], write into cache slots [0]
Step 1:  compute K[1], V[1], write into cache slots [1]
                                                    ▲
                                                    │  one new column per step
         attend Q[t] against cache[:t+1]            │
                                                    │
Step t:  compute K[t], V[t], write into cache slots [t]
```

## 4.2 Applying Buffer Donation

In Notebook 11, we learned that `donate_argnums` on a JIT-compiled function allows XLA to reuse the memory of an input array for the output. This is what makes KV-caching efficient in JAX. 

Without `donate_argnums`, every step of generation would allocate a fresh `(max_seq_len, d_model)` array just to write the single new row!

```{code-cell} ipython3
# A self-contained single-head attention step with KV-cache.
# We work with one attention head, dim = D_MODEL, and a max sequence length.

MAX_SEQ_LEN = 256


def init_kv_cache(max_seq_len, d):
    """Allocate empty K and V buffers."""
    return {
        'k': jnp.zeros((max_seq_len, d)),
        'v': jnp.zeros((max_seq_len, d)),
    }


def attention_step(q_t, k_t, v_t, cache, pos):
    """One decode step: write (k_t, v_t) into the cache at `pos`, attend up to `pos`.

    q_t, k_t, v_t : shape (d_model,) - vectors for the current token only.
    cache         : dict with 'k', 'v' of shape (max_seq_len, d_model).
    pos           : current decoding position (a Python or traced int).

    Returns (output, new_cache).
    """
    # Write the new K, V into the cache buffer.
    new_cache = {
        'k': cache['k'].at[pos].set(k_t),
        'v': cache['v'].at[pos].set(v_t),
    }

    # Attention: q_t (1 × d) against cache K, V (max_seq_len × d).
    # Mask out positions beyond `pos`.
    d = q_t.shape[-1]
    scores = (new_cache['k'] @ q_t) / jnp.sqrt(d)             # (max_seq_len,)
    valid = jnp.arange(new_cache['k'].shape[0]) <= pos
    scores = jnp.where(valid, scores, -jnp.inf)
    attn = jax.nn.softmax(scores)                              # (max_seq_len,)
    output = attn @ new_cache['v']                             # (d_model,)
    return output, new_cache


# JIT with donate_argnums=(3,) so the cache buffer is reused, not reallocated.
attention_step_jit = jax.jit(attention_step, donate_argnums=(3,))

# Demonstrate: decode 8 tokens, threading the cache through each step.
cache = init_kv_cache(MAX_SEQ_LEN, D_MODEL)
key = jax.random.key(7)

for step in range(8):
    key, kq, kk, kv = jax.random.split(key, 4)
    q_t = jax.random.normal(kq, (D_MODEL,))
    k_t = jax.random.normal(kk, (D_MODEL,))
    v_t = jax.random.normal(kv, (D_MODEL,))
    out, cache = attention_step_jit(q_t, k_t, v_t, cache, step)
    print(f"Step {step}: cache filled through position {step}, "
          f"sum |output| = {jnp.sum(jnp.abs(out)).item():.2f}")

print(f"\nCache K shape: {cache['k'].shape}")
print(f"Cache K rows with data: {int(jnp.sum(jnp.any(cache['k'] != 0, axis=1)))}")
print("Each step wrote one row; the buffer's memory was reused thanks to donate_argnums.")
```

---

# 5. End-to-End: Optimizing the Decoder Block

Now we stack the techniques. We take the decoder block and apply, in order:

1. **Baseline**: `float32` params, JIT-compiled on first call.

2. **+ bfloat16 params**: cast params to bf16, run on bf16 input.

3. **+ AOT compilation**: apply `lower().compile()` (from Notebook 11) up front so first-call latency is zero.

Each step is a small, surgical change to what gets fed to `model.apply`. The model definition itself is unchanged.

> **CPU benchmark caveat**: On CPU, bf16 typically runs *slower* than f32 (no native bf16 path), so the speedup column will be flat or negative here. On TPU and on GPUs with bf16 Tensor Cores you'd see the expected 1.5-2× wins. We're showing the *technique*, not the numbers.

```{code-cell} ipython3
def benchmark(fn, *args, n_warmup=3, n_runs=20):
    """Return median execution time in milliseconds."""
    for _ in range(n_warmup):
        out = fn(*args)
        out.block_until_ready()
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        out = fn(*args)
        out.block_until_ready()
        times.append(time.perf_counter() - t0)
    return float(np.median(times)) * 1000.0


x_f32  = jax.random.normal(jax.random.key(99), (SEQ_LEN, D_MODEL))
x_bf16 = x_f32.astype(jnp.bfloat16)

# --- Stage 1: baseline f32 ---
predict_f32 = jax.jit(lambda p, x: model.apply({'params': p}, x))
t_f32 = benchmark(predict_f32, params, x_f32)

# --- Stage 2: bf16 params ---
predict_bf16 = jax.jit(lambda p, x: model.apply({'params': p}, x))
t_bf16 = benchmark(predict_bf16, params_bf16, x_bf16)

# --- Stage 3: bf16 params + AOT-compiled ---
lowered = jax.jit(lambda p, x: model.apply({'params': p}, x)).lower(params_bf16, x_bf16)
predict_aot = lowered.compile()
t_aot = benchmark(predict_aot, params_bf16, x_bf16)

f32_kb  = sum(a.nbytes for a in jax.tree.leaves(params)) / 1024
bf16_kb = sum(a.nbytes for a in jax.tree.leaves(params_bf16)) / 1024

print(f"{'Stage':<28} {'Time (ms)':>10} {'Param mem (KB)':>16}")
print("-" * 60)
print(f"{'Baseline (f32)':<28} {t_f32:>10.3f} {f32_kb:>16.1f}")
print(f"{'+ bf16 params':<28} {t_bf16:>10.3f} {bf16_kb:>16.1f}")
print(f"{'+ AOT compilation':<28} {t_aot:>10.3f} {bf16_kb:>16.1f}")
print()
print("AOT moves compilation cost out of the first request.")
print("bf16 halves the parameter memory; on TPU/GPU it also speeds up the matmuls.")
```

---

# 6. Summary

## What This Notebook Covered

By framing this around a Decoder Block, we demonstrated how the core concepts of JAX come together for production inference:

| Concept (from Notebook 11) | Application in LLM Inference |
|---|---|
| **Precision** | `jax.tree.map(lambda a: a.astype(jnp.bfloat16), params)` to halve memory and boost matmul speed. |
| **Shape Variance** | **Bucketing** - creating a few padded shapes to avoid infinite trace compilation. |
| **Buffer Donation** | **KV-Cache** - using `.at[pos].set(...)` + `donate_argnums` to implement autoregressive decoding without allocations. |
| **Compilation latency** | **AOT Compilation** - using `lower().compile()` to warm up serving binaries before handling traffic. |

## The End of the Tutorial

You made it! From the basics of immutability all the way to autoregressive LLM decoding. You now have the tools to read, write, and optimize modern ML workloads in JAX.

If you are looking for reference implementations to study next:
- **[MaxText](https://github.com/AI-Hypercomputer/maxtext)**: a reference LLM implementation that combines everything covered in Notebooks 01-13 at production scale on TPU.

- **[Flax Examples](https://github.com/google/flax/tree/main/examples)**: a wide variety of domain-specific examples built on Flax.
