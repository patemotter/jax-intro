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

# Part 11: Performance and Profiling

> **Heading for production inference?** Notebook 13 picks up where this notebook ends: int8 quantization, AOT compilation, KV-cache patterns, and serving pipelines.

```{code-cell} ipython3
# @title Setup { display-mode: "form" }

import jax
import jax.numpy as jnp
import jax.lax as lax
import numpy as np
import matplotlib.pyplot as plt
import time
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

plt.style.use('default')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

devices = jax.devices()
n_devices = len(devices)
mesh = Mesh(np.array(devices), axis_names=('data',))

print(f"JAX version:     {jax.__version__}")
print(f"Default backend: {jax.default_backend()}")
print(f"Devices:         {n_devices}")
```

---

# 1. Why JAX Debugging is Different

Three things make JAX debugging unlike standard Python debugging:

1. **Trace-time vs runtime**: Under [`jit`](https://jax.readthedocs.io/en/latest/_autosummary/jax.jit.html), your Python code runs once (tracing) and the compiled result runs many times. Errors and prints behave differently in each phase.

2. **Async dispatch**: JAX operations return immediately while computation continues on the device. Stack traces may point to the dispatch call, not where the error actually occurred.

3. **Functional transformations**: Errors from [`grad`](https://jax.readthedocs.io/en/latest/_autosummary/jax.grad.html), [`vmap`](https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html), and [`jit`](https://jax.readthedocs.io/en/latest/_autosummary/jax.jit.html) often wrap the original error in layers of transformation context, making stack traces harder to read.

> **Rule of thumb**: when something goes wrong, first figure out whether the error is at **trace time** or **runtime**.

+++

---

# 2. Debugging Tools

## 2.1 jax.debug.print

Regular `print` runs at trace time (once). [`jax.debug.print`](https://jax.readthedocs.io/en/latest/_autosummary/jax.debug.print.html) runs at **execution time** (every call).

> **How it works**: `jax.debug.print` inserts a special side-effect callback into the XLA computation graph. The callback is preserved through compilation and fires on the device at execution time, bypassing the usual "no side effects allowed" rule. It's the *only* sanctioned way to emit output from inside a jitted function.

```{code-cell} ipython3
@jax.jit
def debug_demo(x):
    print(f"  [trace-time] x = {x}")          # Only during tracing
    jax.debug.print("  [run-time]   x = {x}", x=x)  # Every execution
    y = x ** 2
    jax.debug.print("  [run-time]   y = {y}", y=y)
    return y

print("--- Call 1 (traces + runs): ---")
r1 = debug_demo(jnp.array(3.0))
print(f"  Result: {r1}")

print("\n--- Call 2 (runs only, no re-trace): ---")
r2 = debug_demo(jnp.array(5.0))
print(f"  Result: {r2}")
```

## 2.2 disable_jit

When you need to debug with regular Python tools (breakpoints, print, pdb), disable JIT temporarily.

```{code-cell} ipython3
@jax.jit
def buggy_function(x):
    intermediate = x * 2
    # Normally you'd use jax.debug.print, but for complex debugging:
    return intermediate + 1

# Disable JIT to use regular Python debugging
with jax.disable_jit():
    print("JIT disabled - all operations run eagerly in Python:")
    result = buggy_function(jnp.array(5.0))
    print(f"  Result: {result}")

print("\nJIT re-enabled outside the context manager.")
print("Use this when you need to step through code with a debugger.")
```

## 2.3 Reading Error Messages

JAX error messages are improving but can still be confusing. Here's how to read the most common ones.

```{code-cell} ipython3
errors = [
    ("TracerBoolConversionError",
     "if traced_value > 0: ...",
     "Use jnp.where() or lax.cond() - Python if can't branch on traced values."),

    ("ConcretizationTypeError",
     "jnp.zeros(traced_int_value)",
     "Array shapes must be known at trace time. Use static_argnames or pad to fixed shapes."),

    ("TypeError: Gradient only defined for scalar...",
     "jax.grad(f) where f returns a vector",
     "grad requires scalar output. Use jax.jacobian for vector outputs, or reduce first."),

    ("UnexpectedTracerError",
     "A tracer (placeholder used inside jit) leaked outside its function",
     "You stored a value from inside a jitted function in a Python global, or returned it through some side channel. Keep all flow through arguments and return values."),

    ("ShapeError / Incompatible shapes",
     "vmap with wrong in_axes or mismatched batch dims",
     "Check that in_axes match the actual array dimensions."),
]

print(f"{'Error':<35s} | {'Common Cause':<45s}")
print("=" * 85)
for name, cause, fix in errors:
    print(f"{name:<35s} | {cause:<45s}")
    print(f"{'':35s} | Fix: {fix}")
    print("-" * 85)
```

## 2.4 NaN Debugging

NaNs can appear in gradients, activations, and losses, then silently propagate through the rest of the computation. JAX has a debug mode that errors at the first operation to produce a NaN, so you can see exactly where it started.

> **Also useful**: [`jax.debug.breakpoint()`](https://jax.readthedocs.io/en/latest/_autosummary/jax.debug.breakpoint.html) drops you into an interactive debugger inside a jitted function. It's powerful but requires a terminal, so we don't demo it inline here.

```{code-cell} ipython3
# When enabled, jax_debug_nans makes JAX error immediately at the operation
# that produced a NaN. Without it, NaNs propagate silently and can be very
# hard to trace back to the offending op.

def naive_log(x):
    return jnp.log(x)  # log of a negative number → NaN

x_bad = jnp.array(-1.0)

# Without jax_debug_nans: NaN appears in the output, no error.
print("Without jax_debug_nans:")
print(f"  log(-1.0) = {naive_log(x_bad)}  (NaN - propagates silently)")

# With jax_debug_nans: the offending operation raises FloatingPointError.
jax.config.update("jax_debug_nans", True)
try:
    naive_log(x_bad)
except FloatingPointError as e:
    print("\nWith jax_debug_nans enabled:")
    print(f"  FloatingPointError caught at the NaN-producing op:")
    print(f"    {str(e)[:160]}")
finally:
    # Turn it off so it doesn't affect later cells (it adds overhead).
    jax.config.update("jax_debug_nans", False)

print("\nIn a real training loop this points you straight at the buggy op")
print("instead of letting the NaN poison every downstream computation.")
```

---

# 3. Profiling JAX Code

## 3.1 Correct Timing

The most common profiling mistake: timing dispatch instead of computation.

+++

## 3.2 Use `block_until_ready()` for Timing

JAX uses **asynchronous dispatch**: it sends work to the hardware and immediately returns control to Python without waiting for the result. When you call a JAX operation, Python gets control back *before the computation finishes*. The result is a future that blocks only when you actually read the value. This is great for throughput, but it makes naive timing wrong.

> **Analogy**: think of it like starting a file download and immediately moving on: the download runs in the background, and you only wait for it when you actually need the file. `block_until_ready()` is the "wait for the download to finish" step.

```{code-cell} ipython3
import time

x = jnp.ones((2000, 2000))

# Warm up first so neither timing includes compilation
y = jnp.dot(x, x); y.block_until_ready()

# WRONG - this measures dispatch time only; JAX returns immediately
# and the matmul keeps running on the device in the background
start = time.time()
y = jnp.dot(x, x)
wrong_time = time.time() - start

# RIGHT - block until the result is actually computed
start = time.time()
y = jnp.dot(x, x)
y.block_until_ready()
right_time = time.time() - start

print(f"Without block_until_ready: {wrong_time*1000:.2f} ms  (just dispatch)")
print(f"With block_until_ready:    {right_time*1000:.2f} ms  (actual compute)")
print("\nThe difference is what JAX did asynchronously while Python moved on.")
```

```{code-cell} ipython3
# @title The Right Way to Benchmark JAX { display-mode: "form" }

x = jax.random.normal(jax.random.key(0), (1000, 1000))

@jax.jit
def matmul(x):
    return x @ x

# Warm up - exclude compilation from benchmarks
_ = matmul(x)
_.block_until_ready()

# WRONG: measures dispatch time only
start = time.time()
y = matmul(x)
wrong = time.time() - start

# RIGHT: includes actual computation
start = time.time()
y = matmul(x)
y.block_until_ready()
right = time.time() - start

print(f"Without block_until_ready: {wrong*1000:.3f} ms  (dispatch only)")
print(f"With block_until_ready:    {right*1000:.3f} ms  (actual computation)")
print(f"\nAlways: warm up → block_until_ready → measure multiple runs → report average.")
```

```{code-cell} ipython3
# @title Robust Benchmarking Function { display-mode: "form" }

def benchmark(fn, *args, n_warmup=3, n_runs=20, label=""):
    """Benchmark a JAX function correctly."""
    # Warm up
    for _ in range(n_warmup):
        result = fn(*args)
        jax.tree.map(lambda x: x.block_until_ready(), result)

    # Time
    times = []
    for _ in range(n_runs):
        start = time.time()
        result = fn(*args)
        jax.tree.map(lambda x: x.block_until_ready(), result)
        times.append(time.time() - start)

    times = np.array(times) * 1000  # ms
    print(f"{label:<40s} {np.median(times):>8.2f} ms (median)  "
          f"±{np.std(times):>6.2f} ms  "
          f"[{np.min(times):.2f}, {np.max(times):.2f}]")
    return times

# Example
x = jax.random.normal(jax.random.key(0), (2000, 2000))
_ = benchmark(jax.jit(lambda x: x @ x), x, label="2000×2000 matmul")
```

## 3.3 Compilation Time vs Execution Time

A function that seems slow may be slow only on the first call (compilation). Measure them separately.

```{code-cell} ipython3
def complex_fn(x):
    for _ in range(10):
        x = jnp.sin(x) + jnp.cos(x)
    return x

x = jax.random.normal(jax.random.key(0), (500, 500))

# Measure compilation + first execution
jit_fn = jax.jit(complex_fn)
start = time.time()
result = jit_fn(x)
result.block_until_ready()
first_call = time.time() - start

# Measure subsequent execution only
start = time.time()
result = jit_fn(x)
result.block_until_ready()
subsequent = time.time() - start

print(f"First call (compile + execute): {first_call*1000:.1f} ms")
print(f"Subsequent call (execute only): {subsequent*1000:.2f} ms")
print(f"Compilation overhead:           {(first_call - subsequent)*1000:.1f} ms")
print(f"\nIf 'slow' means first call only → it's compilation, not computation.")
print(f"If 'slow' persists → it's the computation itself.")
```

---

# 4. Pitfall: Unintentional Recompilation

Every time a jitted function encounters new shapes, dtypes, or static arg values, it recompiles.

**What triggers recompilation:**

| What changed | Recompiles? |
|---|---|
| Different *values* (same shape/dtype) | No |
| Different *shape* | Yes |
| Different *dtype* | Yes |
| Different *static arg value* | Yes |
| Different *device placement* | Yes |

In inference, this typically happens with **variable-length inputs**.

```{code-cell} ipython3
compilation_count = 0

@jax.jit
def detect_recompile(x):
    # This print runs only during tracing → counts compilations
    global compilation_count
    compilation_count += 1
    print(f"  Compiling! (shape={x.shape}, dtype={x.dtype}) - compilation #{compilation_count}")
    return x.sum()

print("--- Same shape: should compile once ---")
detect_recompile(jnp.ones((3, 4)))
detect_recompile(jnp.ones((3, 4)))  # Cached

print("\n--- Different shape: recompiles ---")
detect_recompile(jnp.ones((5, 4)))  # New shape

print("\n--- Different dtype: recompiles ---")
detect_recompile(jnp.ones((3, 4), dtype=jnp.float16))  # New dtype

print(f"\nTotal compilations: {compilation_count}")
```

## 4.1 Static Shape Bucketing

For inference with variable-length inputs, the fix is **bucketing**: pad inputs to one of a few predefined sizes. This limits the number of compilations to the number of buckets.

```{code-cell} ipython3
BUCKETS = [16, 32, 64, 128]

def pad_to_bucket(x, buckets=BUCKETS):
    """Pad sequence length to the next bucket size."""
    seq_len = x.shape[0]
    target_len = min(b for b in buckets if b >= seq_len)
    pad_amount = target_len - seq_len
    # Create padding configuration for each dimension:
    # - Axis 0 (sequence length): pad 0 at start, pad `pad_amount` at end
    # - All other axes (features): pad 0 at both start and end
    pad_config = [(0, pad_amount)] + [(0, 0)] * (x.ndim - 1)
    return jnp.pad(x, pad_config), seq_len

@jax.jit
def process_bucketed(x, actual_len):
    """Process with masking for padded elements."""
    # Create mask for valid elements
    mask = jnp.arange(x.shape[0]) < actual_len
    # Zero out padded elements
    x = jnp.where(mask[:, None], x, 0.0)
    return jnp.sum(x, axis=0)

# Simulate variable-length inputs
print(f"Buckets: {BUCKETS}")
print(f"\n{'Input len':>10} | {'Padded to':>10} | {'Compilations':>13}")
print("-" * 40)

comp_count = [0]
for length in [5, 12, 20, 30, 7, 25, 100]:
    x = jnp.ones((length, 8))
    x_padded, actual_len = pad_to_bucket(x)
    result = process_bucketed(x_padded, actual_len)
    print(f"{length:>10} | {x_padded.shape[0]:>10} | bucket {x_padded.shape[0]}")

print(f"\nOnly {len(BUCKETS)} unique shapes → at most {len(BUCKETS)} compilations,")
print(f"regardless of how many different input lengths you see.")
```

---

# 5. Pitfall: Python Overhead in Hot Loops

Each call to a jitted function has dispatch overhead: Python calls into the JAX runtime, which dispatches to the device. For a training loop calling `train_step` 10,000 times, this overhead adds up.

```{code-cell} ipython3
# Simple example: sum a sequence by repeated addition
x = jax.random.normal(jax.random.key(0), (1000,))

# Approach 1: Python loop calling jitted function
@jax.jit
def add_one(carry, xi):
    return carry + xi

def python_loop_sum(x):
    total = jnp.float32(0.0)
    for xi in x:
        total = add_one(total, xi)
    return total

# Approach 2: scan (one jit call for the entire loop)
@jax.jit
def scan_sum(x):
    total, _ = lax.scan(lambda carry, xi: (carry + xi, None), jnp.float32(0.0), x)
    return total

# Benchmark both
print("Summing 1,000 elements:")
_ = benchmark(python_loop_sum, x, label="Python loop (1000 jit calls)")
_ = benchmark(scan_sum, x, label="scan (1 jit call)")

print("\nscan wins because it eliminates Python dispatch overhead.")
print("The computation is the same; the difference is how many Python→device round-trips.")
```

---

# 6. Pattern: Buffer Donation

When a jitted function receives an input array and produces an output of the same shape, JAX normally allocates **new** memory for the output, copies the result into it, and leaves the input buffer intact. **Buffer donation** tells JAX: "I promise not to use this input after the call, so you can reuse its memory for the output." This eliminates one allocation *and* one copy, which adds up in tight training loops.

> **When to use it**: donate the old `params` to the new `params` in a training step, or donate the old KV-cache to the updated KV-cache during inference. The input and output must have the same shape and dtype.

The requirement: the donated input and the output it maps to must have **the same shape and dtype**: so the memory fits exactly. A common use case is a parameter update step that takes old params and returns new params of the same shape. Donating the old params means JAX writes new params into the same memory, avoiding a 2× peak allocation.

```{code-cell} ipython3
# A pure update function. Donation is configured at jit time via donate_argnums,
# not by changing the function body.
def update_params(params, grads, lr=0.01):
    return jax.tree.map(lambda p, g: p - lr * g, params, grads)

# Without donation: each call allocates fresh memory for new_params.
update_no_donate = jax.jit(update_params)

# With donation: donate_argnums=(0,) tells JAX argument 0 (params) is dead after the call,
# so its buffer can be reused for the output. Peak memory drops from ~2× params to ~1×.
update_donated = jax.jit(update_params, donate_argnums=(0,))

# Demo
key = jax.random.key(0)
model_params = {'w': jnp.ones((2, 2)), 'b': jnp.ones((2,))}
fake_grads = jax.tree.map(lambda x: jnp.ones_like(x) * 0.01, model_params)

param_bytes = sum(x.nbytes for x in jax.tree.leaves(model_params))
print(f"Parameter size: {param_bytes / 1024:.2f} KB")
print()
print("Without donation: each update allocates new memory for new_params")
print("  → Peak usage ≈ 2× param size (old + new both live simultaneously)")
print()
print("With donate_argnums=(0,): old params memory is reused in-place")
print("  → Peak usage ≈ 1× param size (no extra allocation)")
print()
print("For this tiny model the savings are negligible. For a model with billions")
print("of parameters, avoiding the 2× peak is essential.")

# After donation, the donated buffer is consumed - don't read model_params after this call.
new_params = update_donated(model_params, fake_grads)
print(f"\nnew_params created: {jax.tree.leaves(new_params)[0].shape}")
print("(model_params has been donated - using it again is undefined behavior.)")
```

---

# 7. Precision and Quantization

Inference is usually **memory-bandwidth bound**, not compute-bound. The biggest single lever is moving fewer bytes per inference.

## 7.1 bfloat16 vs float32

bfloat16 has the same exponent range as float32 (so it handles the same magnitudes) but less mantissa precision. On modern GPU/TPU accelerators, bf16 operations are typically 1.5-2× faster than f32 *and* use half the memory. On plain CPU the matmul timings below will look roughly even (CPU lacks a native bf16 path), but the memory savings still apply.

> **Why bfloat16 instead of float16?** Both are 16-bit, but they use those bits differently. float16 has more mantissa bits (higher precision) but a smaller exponent range, so gradients can underflow to zero during training, a common instability. bfloat16 keeps float32's full 8-exponent bits, so it handles the same range of values and is far less prone to overflow/underflow. For ML workloads, bfloat16 is almost always the right choice.

> **CPU caveat**: On CPU, bf16 ops often run *slower* than f32 because CPUs lack native bf16 acceleration. The wins shown here apply to TPU and GPUs with bf16 Tensor Cores.

```{code-cell} ipython3
# Demonstrate precision differences
x_f32 = jnp.array(1.0, dtype=jnp.float32)
x_bf16 = jnp.array(1.0, dtype=jnp.bfloat16)
x_f16 = jnp.array(1.0, dtype=jnp.float16)

print(f"{'dtype':<10} | {'bytes':>6} | {'max value':>15} | {'1/3':>15}")
print("-" * 55)
for name, x in [("float32", x_f32), ("bfloat16", x_bf16), ("float16", x_f16)]:
    max_val = jnp.finfo(x.dtype).max
    third = x / 3
    print(f"{name:<10} | {x.itemsize:>6} | {max_val.item():>15.2e} | {third.item():.10f}")

print("\nbfloat16: same range as float32, less precision - ideal for ML")
print("float16:  smaller range (can overflow), more precision than bf16")
```

```{code-cell} ipython3
# Benchmark bf16 vs f32
n = 1024
x_f32 = jax.random.normal(jax.random.key(0), (n, n))
w_f32 = jax.random.normal(jax.random.key(1), (n, n))
x_bf16 = x_f32.astype(jnp.bfloat16)
w_bf16 = w_f32.astype(jnp.bfloat16)

matmul_f32 = jax.jit(lambda x, w: x @ w)
matmul_bf16 = jax.jit(lambda x, w: x @ w)

_ = benchmark(matmul_f32, x_f32, w_f32, label=f"float32 {n}×{n} matmul")
_ = benchmark(matmul_bf16, x_bf16, w_bf16, label=f"bfloat16 {n}×{n} matmul")

# Check accuracy
result_f32 = matmul_f32(x_f32, w_f32)
result_bf16 = matmul_bf16(x_bf16, w_bf16).astype(jnp.float32)
rel_error = jnp.mean(jnp.abs(result_f32 - result_bf16) / (jnp.abs(result_f32) + 1e-8))
print(f"\nRelative error (bf16 vs f32): {rel_error.item():.6f}")
print(f"Memory: f32 uses {x_f32.nbytes/1024:.0f} KB, bf16 uses {x_bf16.nbytes/1024:.0f} KB")
```

## 7.2 int8 Quantization

bfloat16 gets you 2× memory savings vs `float32`. **int8 quantization** goes further, 4× vs `float32`, by storing each weight as a 1-byte integer plus a small floating-point scale factor.

The basic idea for a weight matrix `W` of shape `(d_in, d_out)`. int8 values range from -128 to 127; we use 127 as the symmetric maximum:

```
For each output channel j (a column of W):
    scale[j]  = max(|W[:, j]|) / 127           # a single float32
    W_int8[:, j] = round(W[:, j] / scale[j])   # int8 values in [-127, 127]
```

At inference time you dequantize on the fly: `W_approx[i, j] = W_int8[i, j] * scale[j]`.

```{code-cell} ipython3
def quantize_weights(w_f32):
    """Quantize a (d_in, d_out) weight matrix to int8 with per-output-channel scales."""
    max_per_col = jnp.max(jnp.abs(w_f32), axis=0)        # (d_out,)
    scale = max_per_col / 127.0                            # (d_out,)
    safe_scale = jnp.where(scale == 0, 1.0, scale)
    w_int8 = jnp.round(w_f32 / safe_scale).astype(jnp.int8)
    return w_int8, scale


def dequant_matmul(x, w_int8, scale):
    """Matmul with on-the-fly dequantization."""
    w_approx = w_int8.astype(x.dtype) * scale             # broadcast (d_in, d_out) × (d_out,)
    return x @ w_approx

w_f32 = jax.random.normal(jax.random.key(2), (1024, 1024))
w_int8, scale = quantize_weights(w_f32)

x = jax.random.normal(jax.random.key(3), (128, 1024))
y_f32  = x @ w_f32
y_int8 = dequant_matmul(x, w_int8, scale)
rel_err = jnp.mean(jnp.abs(y_f32 - y_int8) / (jnp.abs(y_f32) + 1e-6))

f32_kb  = w_f32.nbytes / 1024
int8_kb = (w_int8.nbytes + scale.nbytes) / 1024
print(f"Weight matrix shape:   {w_f32.shape}")
print(f"  float32 size:        {f32_kb:.2f} KB")
print(f"  int8 + scale size:   {int8_kb:.2f} KB  ({int8_kb / f32_kb:.0%} of float32)")
print(f"\nRelative error after dequant matmul: {rel_err.item():.4f}")
```

---

# 8. AOT Compilation and Export

By default JIT compiles on the **first call**. For a long-running server that's a single slow first request, then everything is fast. For batch serving or low-latency requirements, you may want to compile **ahead of time**: separately from request handling, and hand the compiled artifact to your serving code.

## 8.1 The Three-Step Lifecycle

Behind the scenes, `jax.jit` does three things:

```
Python code  ──►  StableHLO  ──►  Compiled binary  ──►  Run
                  (lowering)      (XLA compilation)
```

You can run them as separate steps:

```{code-cell} ipython3
@jax.jit
def complex_fn(x, y):
    return jnp.sin(x) @ jnp.cos(y)

x = jax.random.normal(jax.random.key(11), (512, 512))
y = jax.random.normal(jax.random.key(12), (512, 512))

# Step 1: Lower - trace and convert to StableHLO
t0 = time.perf_counter()
lowered = complex_fn.lower(x, y)
t1 = time.perf_counter()

# Step 2: Compile - run the XLA compiler
compiled = lowered.compile()
t2 = time.perf_counter()

# Step 3: Execute - the cached compiled artifact
out = compiled(x, y)
out.block_until_ready()
t3 = time.perf_counter()

print(f"Lowering:         {(t1-t0)*1000:>7.1f} ms  (Python → StableHLO)")
print(f"Compilation:      {(t2-t1)*1000:>7.1f} ms  (StableHLO → XLA binary)")
print(f"First execution:  {(t3-t2)*1000:>7.1f} ms")
```

## 8.2 jax.export

`jax.export` packages a function as an artifact containing its StableHLO and input/output type signatures. The artifact is platform-independent and can be saved to disk, shipped to a serving binary, and re-compiled on the target machine.

```{code-cell} ipython3
exported = jax.export.export(complex_fn)(x, y)

print(f"Exported function:")
print(f"  Function name:  {exported.fun_name}")
print(f"  Input shapes:   {[aval.shape for aval in exported.in_avals]}")
print(f"  Output shapes:  {[aval.shape for aval in exported.out_avals]}")

try:
    serialized = exported.serialize()
    restored = jax.export.deserialize(serialized)
    out_restored = restored.call(x, y)
    print(f"\nSerialized size: {len(serialized) / 1024:.1f} KB")
except ImportError:
    print("\nSerialization demo skipped - install `flatbuffers` to enable it.")
```

---

# 9. Summary

## Key Takeaways

- **JAX debugging is different**: Python's `pdb` doesn't work inside `jit`. Use `jax.debug.print`, `disable_jit`, and `jax_debug_nans` instead.

- **Time computation, not dispatch**: always `block_until_ready()` before stopping the clock, and warm up to exclude compilation.

- **Recompilation is the silent performance killer**: every new shape or dtype triggers a fresh trace and compile. For variable-length inputs, pad to fixed sizes.

- **Buffer donation** (`donate_argnums`) eliminates one memory allocation per training step, essential for large models.

- **Precision and Quantization**: `bfloat16` and `int8` cut memory drastically while preserving necessary precision for ML workloads.

- **AOT Compilation**: Use `jax.jit(f).lower().compile()` to eliminate first-call compilation latency, and `jax.export` to serialize portable models.

## What's Next

In **Notebook 12: Pallas Kernels**, we'll go one level deeper: writing custom hardware kernels with Pallas when XLA's automatic fusion isn't enough.

If you're heading toward production inference, **Notebook 13: Capstone - LLM Inference** applies everything learned across the series into a full implementation of an autoregressive Decoder block.

+++

---

# 10. Exercises

1. **Correct benchmarking**: Pick any of your own JAX functions. Time it once without `block_until_ready` and once with. Then add `n_warmup=3` and report the median over 20 runs.

2. **Recompilation hunt**: Define a jitted function with a `print` statement at the top of its body. Call it with several different input shapes. Count how many times it traces.

3. **Buffer donation**: Take a jitted parameter-update step (params in, new params out, same shapes). Add `donate_argnums=(0,)`. Verify the output is correct and explain what happens if you try to use the old `params` variable after the donated call.

4. **AOT Verification**: Lower and compile your function from exercise 1. Call the compiled artifact multiple times and verify the compilation time is indeed 0.
