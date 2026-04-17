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

# Part 4: JIT and Vmap

```{code-cell} ipython3
# @title Setup { display-mode: "form" }

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('default')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

print(f"JAX version:     {jax.__version__}")
print(f"Devices:         {jax.devices()}")
print(f"Default backend: {jax.default_backend()}")
```

```{code-cell} ipython3
# @title MLP and Loss Utilities (used throughout this notebook) { display-mode: "form" }

def init_mlp_params(key, layer_sizes):
    """Initialize MLP parameters (weights scaled by sqrt(2/n_in))."""
    params = []
    for i in range(len(layer_sizes) - 1):
        key, w_key = jax.random.split(key)
        n_in = layer_sizes[i]
        n_out = layer_sizes[i + 1]
        w = jax.random.normal(w_key, (n_in, n_out)) * jnp.sqrt(2.0 / n_in)
        b = jnp.zeros(n_out)
        params.append((w, b))
    return params

def mlp_forward(params, x):
    """Forward pass through an MLP."""
    for i, (w, b) in enumerate(params):
        x = x @ w + b
        if i < len(params) - 1:
            x = jnp.maximum(x, 0)
    return x

def mse_loss(params, x, y):
    """Scalar MSE loss for a single example (batched version used later)."""
    pred = mlp_forward(params, x)
    return jnp.mean((pred - y) ** 2)

# Same architecture and seed used across notebooks
key = jax.random.key(42)
layer_sizes = [8, 64, 32, 1]
params = init_mlp_params(key, layer_sizes)

print("MLP reconstructed: ", " → ".join(str(s) for s in layer_sizes))
```

---

# 1. JIT Compilation

Without [`jit`](https://jax.readthedocs.io/en/latest/_autosummary/jax.jit.html), every JAX operation dispatches individually to the backend: one Python → XLA round trip per operation. With [`jit`](https://jax.readthedocs.io/en/latest/_autosummary/jax.jit.html), JAX **traces** your function once, compiles the entire computation graph, and executes the optimized version on subsequent calls.

```
  Without jit: each op round-trips through Python

     Python ──► XLA ──► op_1 ──► Python
     Python ──► XLA ──► op_2 ──► Python
     Python ──► XLA ──► op_3 ──► Python
     ...
     Cost = N operations × dispatch overhead

  With jit: one call runs the entire compiled graph

     Python ──► XLA ──► [ op_1 → op_2 → op_3 → ... fused ] ──► Python
     Cost = 1 dispatch + fused execution
```

> **jit doesn't change what your function computes. It changes how fast it computes it.**

```{code-cell} ipython3
import time

# Generate test data
key = jax.random.key(0)
x_test = jax.random.normal(key, (8,))

# Non-jitted
def train_step_nojit(params, x, y, lr=0.01):
    loss, grads = jax.value_and_grad(mse_loss)(params, x, y)
    new_params = [(w - lr * dw, b - lr * db) for (w, b), (dw, db) in zip(params, grads)]
    return new_params, loss

# Jitted
@jax.jit
def train_step_jit(params, x, y, lr=0.01):
    loss, grads = jax.value_and_grad(mse_loss)(params, x, y)
    new_params = [(w - lr * dw, b - lr * db) for (w, b), (dw, db) in zip(params, grads)]
    return new_params, loss

y_test = jnp.array(1.0)

# Warm up (first call includes compilation)
_ = train_step_nojit(params, x_test, y_test)
_ = train_step_jit(params, x_test, y_test)

# Benchmark
n_iters = 500
start = time.time()
p = params
for _ in range(n_iters):
    p, _ = train_step_nojit(p, x_test, y_test)
nojit_time = time.time() - start

start = time.time()
p = params
for _ in range(n_iters):
    p, _ = train_step_jit(p, x_test, y_test)
# Block here (after the loop) so async dispatch doesn't under-count the last iterations
jax.tree.map(lambda x: x.block_until_ready(), p)
jit_time = time.time() - start

print(f"{'Method':<20} | {'Time':>10} | {'Per step':>12}")
print("-" * 50)
print(f"{'Without jit':<20} | {nojit_time:>8.3f} s | {nojit_time/n_iters*1000:>9.3f} ms")
print(f"{'With jit':<20} | {jit_time:>8.3f} s | {jit_time/n_iters*1000:>9.3f} ms")
print(f"{'Speedup':<20} | {nojit_time/jit_time:>9.1f}x |")
print()
print("Note: a 100×+ speedup is dramatic because the per-step work is tiny -")
print("Python dispatch overhead dominates. Realistic GPU/TPU training-step")
print("speedups are usually in the 2-20× range.")
```

---

# 2. How Tracing Works

When you first call a jitted function, JAX feeds **tracer objects** (abstract placeholders that carry shape and dtype but no values) through your function. It records every operation to build a computation graph (called a **jaxpr**), then hands that graph to XLA for compilation.

The compiled version is cached. Subsequent calls with the same shapes and dtypes skip tracing and compilation entirely.

```text
  First call:

  ┌──────────────────┐
  │ Python JAX code  │
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────────────────────────┐
  │ Feed abstract tracers (shape/dtype)  │
  └────────┬─────────────────────────────┘
           │
           ▼
  ┌──────────────────────────────────────┐
  │ Record operations into jaxpr         │
  └────────┬─────────────────────────────┘
           │
           ▼
  ┌──────────────────────────────────────┐
  │ XLA Compiler                         │
  └────────┬─────────────────────────────┘
           │
           ▼
  ┌──────────────────────────────────────┐
  │ Compiled binary (cached)             │
  └────────┬─────────────────────────────┘
           │
           ▼
  ┌──────────────────────────────────────┐
  │ Run on device                        │
  └──────────────────────────────────────┘

  Subsequent calls (same shapes/dtypes):

  ┌──────────────────┐      ┌──────────────────────────┐      ┌───────────────┐
  │ Python JAX code  ├─────►│ Compiled binary (cached) ├─────►│ Run on device │
  └──────────────────┘      └──────────────────────────┘      └───────────────┘
```

+++

Your Python function only runs **once** during tracing. After that, JAX runs the compiled version directly on the device. This is why prints inside `@jax.jit` functions only fire on the first call. Everything after the first call bypasses Python entirely.

```{code-cell} ipython3
@jax.jit
def demo_tracing(x):
    print(f"  TRACING with x = {x}")  # This runs during tracing only
    result = x * 2 + 1
    return result

print("--- First call (traces + compiles): ---")
r1 = demo_tracing(jnp.array(5.0))
print(f"  Result: {r1}")

print("\n--- Second call (uses cached compilation): ---")
r2 = demo_tracing(jnp.array(10.0))
print(f"  Result: {r2}")

print("\n--- Third call with DIFFERENT SHAPE (re-traces): ---")
r3 = demo_tracing(jnp.array([1.0, 2.0]))
print(f"  Result: {r3}")

print("\nThe print ran during tracing (calls 1 and 3), not during cached execution (call 2).")
print("Notice: during tracing, x shows up as a JitTracer(...) placeholder, not a concrete value.")
```

## 2.1 make_jaxpr: Reading JAX's IR

[`jax.make_jaxpr`](https://jax.readthedocs.io/en/latest/_autosummary/jax.make_jaxpr.html) shows you the computation graph that JIT would compile. It's the most useful debugging tool for understanding what JAX sees.

**How to read a jaxpr output:**
- Each `let a = op(b, c)` line is one XLA primitive operation

- The names (`a`, `b`, `c`) are abstract tracers: placeholders with shape and dtype, not values

- The final `in [...]` line lists the outputs of the function

You don't need to fully understand jaxpr syntax; just look for which operations appear and in what order.

```{code-cell} ipython3
def simple_fn(x, y):
    z = x * y + 1
    return jnp.sin(z)

print("jaxpr for simple_fn(float32, float32):")
print(jax.make_jaxpr(simple_fn)(jnp.float32(1.0), jnp.float32(2.0)))

print("\nReading this:")
print("  a:f32[] = mul(x, y)         ← z = x * y")
print("  b:f32[] = add(a, 1.0)       ← z + 1")
print("  c:f32[] = sin(b)            ← sin(z + 1)")
print("  return c")
print("\nNo Python control flow, no side effects - just the math.")
```

---

# 3. When JIT Breaks

JIT has strict rules. When you violate them, you get `TracerError`, JAX's way of saying "I can't trace this."

```
  Concrete value                   Abstract tracer
  (normal Python)                  (what JIT sees during tracing)

  x = 5.0                          x = ShapedArray(float32[])
  x > 0   →   True                 x > 0   →   ShapedArray(bool[])
                                       (shape known, value unknown)

  if True: ...                     if ShapedArray(...):  ← ERROR
  works                            Python can't branch on a tracer
```

## 3.1 Common TracerError Causes

### Cause 1: Data-Dependent Control Flow

During tracing, `x` is an **abstract placeholder** (a tracer object): it carries shape and dtype but has no concrete value. So `x > 0` returns another tracer, not `True` or `False`. Python's `if` statement can't branch on a tracer.

```{code-cell} ipython3
# This FAILS - the if/else depends on the VALUE of x, which is unknown at trace time
def bad_relu(x):
    if x > 0:       # Python if on a traced value → TracerError
        return x
    else:
        return 0.0

try:
    jax.jit(bad_relu)(jnp.array(1.0))
except jax.errors.TracerBoolConversionError as e:
    print(f"Error: {type(e).__name__}")
    print(f"  {str(e)[:200]}...")

# Fix: use jnp.where (always evaluates both branches)
def good_relu(x):
    return jnp.where(x > 0, x, 0.0)

print(f"\nFixed with jnp.where: {jax.jit(good_relu)(jnp.array(1.0))}")
print(f"Also works for negative: {jax.jit(good_relu)(jnp.array(-1.0))}")
```

### Cause 2: Dynamic Shapes

JAX traces with **fixed shapes**. You can't create arrays whose size depends on input values.

```{code-cell} ipython3
# This FAILS - jnp.zeros(n) where n is traced
def bad_dynamic(n):
    return jnp.zeros(n)  # n must be known at trace time

try:
    jax.jit(bad_dynamic)(3)
except Exception as e:
    print(f"Error: {type(e).__name__}")
    print(f"  {str(e)[:200]}...")

# Fix: mark n as static, so a fresh version is compiled for each value of n
# (and inside the compiled version, n is a concrete Python int again).
fixed_dynamic = jax.jit(bad_dynamic, static_argnames=('n',))
print(f"\nFixed with static_argnames: {fixed_dynamic(3)}")
print(f"Calling with a different n recompiles: {fixed_dynamic(5)}")
```

### Cause 3: Side Effects

Side effects don't raise `TracerError`; they produce silently wrong results, which is worse. Recall from [Notebook 01](01_jax_arrays_and_immutability.ipynb): if your function mutates external state, calls `np.random` (non-JAX), or Python-`print`s a computed value, those actions happen at **trace time** and get baked into the compiled graph. Subsequent calls don't re-run the side effect; they just reuse the cached result.

The fix is to pass all state through function arguments and return values, and to use [`jax.debug.print`](https://jax.readthedocs.io/en/latest/_autosummary/jax.debug.print.html) when you need a print that runs at every call.

+++

## 3.2 static_argnames: The Escape Hatch

When an argument must be known at trace time (because it determines shapes or control flow), mark it as static. JAX will re-trace and re-compile whenever that argument changes.

> **Best Practice**: JAX documentation prefers `static_argnames` over the older `static_argnums`. Using argument names instead of positional indices makes the code much more robust to signature changes (e.g., adding a new parameter or swapping argument order).

> **Warning**: If a static argument takes many different values, you'll compile many versions. This is a compilation cache explosion and it will make your program slow. For example, if you mark `batch_size` as static and call with 8, 16, 24, 32, 64 … JAX compiles a fresh version for each, potentially dozens or hundreds of compilations that accumulate silently.

```{code-cell} ipython3
def add_ones(x, n):
    print(f"  Tracing with n={n}")  # Only runs during tracing
    return x + jnp.ones(n)

# n must be static because it determines the shape of jnp.ones(n)
add_ones_static = jax.jit(add_ones, static_argnames=('n',))

print("--- Same n: traces once, then cached ---")
print(f"Result: {add_ones_static(jnp.array([1.0, 2.0, 3.0]), 3)}")
print(f"Result: {add_ones_static(jnp.array([4.0, 5.0, 6.0]), 3)}")

print("\n--- Different n: must re-trace ---")
print(f"Result: {add_ones_static(jnp.array([1.0, 2.0]), 2)}")

print("\nEach unique value of n triggers a new compilation.")
```

## 3.3 What Triggers Recompilation

| Change | Recompiles? |
|---|---|
| Different **values** (same shape/dtype) | No - uses cached version |
| Different **shape** | Yes |
| Different **dtype** | Yes |
| Different **static arg value** | Yes |
| Different **number of args** | Yes |

This is why `block_until_ready()` matters for benchmarking: the first call includes compilation time, subsequent calls don't.

+++

---

# 4. The Problem vmap Solves

Our `mlp_forward` from Notebook 03 (re-defined in the setup cell above) processes one input at a time. To run it on a batch of inputs, the obvious approach is a Python `for` loop:

```python
preds = jnp.array([mlp_forward(params, xi) for xi in x_batch])
```

This is slow: it loops in Python and dispatches one operation per element to the device.

```
  Python loop: N separate dispatches, one op at a time

     for xi in batch:         ┌──► mlp_forward(xi) ──► y_0
        preds.append(f(xi))   ├──► mlp_forward(xi) ──► y_1
                              ├──► mlp_forward(xi) ──► y_2
                              └──► ... (N times)

  vmap: one batched op - XLA sees the whole batch

     vmap(f)(batch)  ──►  [ batched mlp_forward ]  ──►  [y_0 ... y_N]
                           (fused, vectorized, one dispatch)
```

There are three ways to handle batching. Let's compare them.

```{code-cell} ipython3
key = jax.random.key(0)
x_batch = jax.random.normal(key, (1000, 8))

# --- Approach 1: Python loop ---
def batch_loop(params, x_batch):
    return jnp.array([mlp_forward(params, xi) for xi in x_batch])

# --- Approach 2: Manual batching (rewrite forward pass for batches) ---
def mlp_forward_batched(params, x):
    """Forward pass rewritten to handle batch dimension."""
    for i, (w, b) in enumerate(params):
        x = x @ w + b  # Works because @ broadcasts over batch dim
        if i < len(params) - 1:
            x = jnp.maximum(x, 0)
    return x

# --- Approach 3: vmap ---
batch_vmap = jax.vmap(mlp_forward, in_axes=(None, 0))
# in_axes=(None, 0) means: don't map over params, map over axis 0 of x

# Warm up JIT
batch_loop_jit = jax.jit(batch_loop)
batch_manual_jit = jax.jit(mlp_forward_batched)
batch_vmap_jit = jax.jit(batch_vmap)

_ = batch_loop_jit(params, x_batch)
_ = batch_manual_jit(params, x_batch)
_ = batch_vmap_jit(params, x_batch)

# Time them
results = {}
for name, fn in [("Python loop", batch_loop_jit),
                 ("Manual batch", batch_manual_jit),
                 ("vmap", batch_vmap_jit)]:
    start = time.time()
    for _ in range(100):
        out = fn(params, x_batch)
        out.block_until_ready()
    elapsed = time.time() - start
    results[name] = elapsed / 100

print(f"{'Method':<20} | {'Time (ms)':>10} | {'vs vmap':>10}")
print("-" * 48)
vmap_time = results["vmap"]
for name, t in results.items():
    print(f"{name:<20} | {t*1000:>8.3f} ms | {t/vmap_time:>8.1f}x")

# Verify all give the same result
out_loop = batch_loop_jit(params, x_batch)
out_manual = batch_manual_jit(params, x_batch)
out_vmap = batch_vmap_jit(params, x_batch)
print(f"\nAll equal? loop≈manual: {jnp.allclose(out_loop.squeeze(), out_manual.squeeze(), atol=1e-5)}, "
      f"manual≈vmap: {jnp.allclose(out_manual.squeeze(), out_vmap.squeeze(), atol=1e-5)}")
```

The Python loop is slow because it can't be fully optimized by XLA. Manual batching works but requires rewriting your function. **vmap gives you the speed of manual batching with the clarity of writing for a single example.**

> **Why is vmap faster than a Python loop?** A Python loop dispatches one operation at a time to the device. `vmap` rewrites the function to include the batch dimension *before* it ever reaches XLA, so XLA sees the entire batch at once and can fuse, vectorize, and schedule it optimally.

> **vmap transforms a function that works on one example into a function that works on a batch, automatically.**

+++

---

# 5. vmap Basics: in_axes and out_axes

[`vmap`](https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html) needs to know **which arguments to map over** and **along which axis**. That's what `in_axes` controls.

| `in_axes` value | Meaning |
|---|---|
| `0` | Map over axis 0 (batch dimension) |
| `1` | Map over axis 1 |
| `None` | Don't map - broadcast this argument to all batch elements |
| Tuple | Per-argument specification, e.g., `(0, None)` |

`out_axes` works symmetrically for *outputs*: it controls which axis of each output array the batch results are stacked along. The default `0` is almost always correct. Use `out_axes=1` if you want the batch dimension to be the second axis in the output.

```{code-cell} ipython3
# Our forward pass takes (params, x_single)
# We want to map over x (axis 0) but broadcast params (None)
batched_forward = jax.vmap(mlp_forward, in_axes=(None, 0))

# Now it handles batches natively
x_batch = jax.random.normal(jax.random.key(0), (5, 8))
preds = batched_forward(params, x_batch)
print(f"Input batch shape:  {x_batch.shape}")
print(f"Output batch shape: {preds.shape}")

# Compare with single-example calls
for i in range(3):
    single = mlp_forward(params, x_batch[i])
    print(f"  Sample {i}: vmap={preds[i].item():.6f}, single={single.item():.6f}, "
          f"match={jnp.isclose(preds[i], single)}")
```

```{code-cell} ipython3
# Multi-output example: vmap handles tuple/dict returns naturally.
# By default it stacks each output along axis 0.

def compute_stats(x):
    """For a single vector, return (mean, std)."""
    return jnp.mean(x), jnp.std(x)

x_batch = jax.random.normal(jax.random.key(0), (100, 8))

# Map over axis 0, stack results along axis 0 (default out_axes=0)
means, stds = jax.vmap(compute_stats)(x_batch)
print(f"Default out_axes=0:")
print(f"  Means shape: {means.shape}, stds shape: {stds.shape}")

# out_axes=1 instead would stack along axis 1 (transposed layout).
# Here that would just turn (100,) into (100,) since each output is scalar,
# but for multi-element outputs it controls where the batch dim ends up.
def double_then_negate(x):
    return x * 2, -x  # both outputs have the same shape as x

x2 = jax.random.normal(jax.random.key(1), (5, 3))   # 5 vectors of length 3
d_stack0, n_stack0 = jax.vmap(double_then_negate, out_axes=0)(x2)
d_stack1, n_stack1 = jax.vmap(double_then_negate, out_axes=1)(x2)
print(f"\nout_axes=0: doubled.shape = {d_stack0.shape}  (batch on axis 0)")
print(f"out_axes=1: doubled.shape = {d_stack1.shape}  (batch on axis 1)")
```

---

# 6. vmap Patterns

## 6.1 vmap + grad: Per-Example Gradients

A useful composition: compute the gradient for **each example independently**, rather than averaging across the batch like a normal training step. This is handy when you want to inspect each sample's contribution, for example to find which examples have unusually large gradients, or to clip each one separately before averaging.

```{code-cell} ipython3
def single_loss(params, x, y):
    """Loss for a SINGLE example."""
    pred = mlp_forward(params, x)
    return jnp.mean((pred - y) ** 2)

# grad w.r.t. params for a single example
single_grad_fn = jax.grad(single_loss)

# vmap over examples: per-example gradients
per_example_grad_fn = jax.vmap(single_grad_fn, in_axes=(None, 0, 0))

# Generate data
key = jax.random.key(0)
x_batch = jax.random.normal(key, (32, 8))
y_batch = jax.random.normal(jax.random.key(1), (32,))

per_example_grads = per_example_grad_fn(params, x_batch, y_batch)

# Each gradient has a batch dimension
print("Per-example gradient shapes:")
for i, (dw, db) in enumerate(per_example_grads):
    print(f"  Layer {i}: dw {dw.shape}, db {db.shape}")

print(f"\nFirst dimension (32) = number of examples.")
print(f"Each example has its own independent gradient.")
```

## 6.2 vmap + jit: The Standard Pattern

Wrapping `jit` around `vmap` is the typical recipe for batched-and-compiled code: `vmap` adds the batch dimension, `jit` compiles the whole thing.

```{code-cell} ipython3
# The standard pattern: vmap for batching, jit for speed
batched_forward_fast = jax.jit(jax.vmap(mlp_forward, in_axes=(None, 0)))

x_large = jax.random.normal(jax.random.key(0), (10000, 8))

# Warm up
_ = batched_forward_fast(params, x_large)

start = time.time()
for _ in range(100):
    out = batched_forward_fast(params, x_large)
    out.block_until_ready()
elapsed = (time.time() - start) / 100

print(f"jit(vmap(forward)) on 10,000 examples: {elapsed*1000:.3f} ms (wall clock)")
print("Per-call cost stays low even at large batch sizes because the work is one fused kernel.")
```

## 6.3 Nested vmap: Multiple Batch Dimensions

You can stack vmap calls to handle multiple batch dimensions. A common use case: computing pairwise distances between two sets of points.

```
  Inner vmap: fix A[i], map over all B[j]    → row of distances
  Outer vmap: do that for every A[i]         → full matrix

                       B[0]          B[1]
                     ┌────────────────────────────┐
                A[0] │ d(A[0],B[0])  d(A[0],B[1]) │
                A[1] │ d(A[1],B[0])  d(A[1],B[1]) │
                A[2] │ d(A[2],B[0])  d(A[2],B[1]) │
                     └────────────────────────────┘
                             shape (3, 2)
```

```{code-cell} ipython3
def single_distance(x, y):
    """Euclidean distance between two vectors."""
    return jnp.sqrt(jnp.sum((x - y) ** 2))

# Pairwise: for every x_i and every y_j, compute distance(x_i, y_j).
# Step 1: For one x, get distances to every y. Hold x fixed (in_axes=None),
#         map over y's axis 0.
distance_one_x_to_all_y = jax.vmap(single_distance, in_axes=(None, 0))
# Step 2: Do that for every x. Map over x's axis 0, hold the y array fixed.
pairwise_distance = jax.vmap(distance_one_x_to_all_y, in_axes=(0, None))

# Test
a = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])  # 3 points
b = jnp.array([[1.0, 1.0], [2.0, 2.0]])                 # 2 points

D = pairwise_distance(a, b)
print(f"Points A: {a.shape[0]}, Points B: {b.shape[0]}")
print(f"Distance matrix shape: {D.shape}")
print(f"Distance matrix:\n{D}")
print(f"\nD[0,0] = dist([0,0], [1,1]) = √2 = {jnp.sqrt(2.0):.4f}  ✓")
```

---

# 7. Summary

## Key Takeaways

- **`jax.jit` compiles for performance**: it traces your Python code and compiles it via XLA into optimized device code.

- **Tracing uses abstract shapes**: the JIT compiler only knows the shape and dtype of your arrays, not their concrete values. Data-dependent Python control flow (`if x.sum() > 0:`) breaks compilation.

- **Use `static_argnames` for structural inputs**: if an argument changes the structure of the computation (a boolean flag, a shape parameter), mark it static to force recompilation when it changes.

- **`jax.vmap` eliminates manual batching**: write your function for a single example, then `vmap` it to handle batches. `in_axes` controls which arguments are mapped and which are broadcast.

## What's Next

In **Notebook 05: Pytrees**, we'll learn how JAX transformations propagate through nested structures of dicts/lists/tuples: the foundation that lets `grad` and `tree.map` operate on entire model parameter sets at once.

+++

---

# 8. Exercises

1. **in_axes exploration**: Write a function `f(x, scale)` and use `vmap` to map over `x` while broadcasting `scale`. Then flip it: map over `scale` and broadcast `x`.

2. **Vector normalization**: Write a function that normalizes a 1D vector (subtracts the mean). Use `vmap` to apply this function to every row of a 2D matrix.

3. **Tracing count**: Define a jitted function that prints a message at the top. Call it three times with the same shape, then once with a new shape. Count how many times the print fires and explain why.
