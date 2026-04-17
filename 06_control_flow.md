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

# Part 6: Control Flow

```{code-cell} ipython3
# @title Setup { display-mode: "form" }

import jax
import jax.numpy as jnp
import jax.lax as lax
import numpy as np
import matplotlib.pyplot as plt
import time

plt.style.use('default')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

print(f"JAX version:     {jax.__version__}")
print(f"Devices:         {jax.devices()}")
print(f"Default backend: {jax.default_backend()}")
```

```{code-cell} ipython3
# @title Reconstruct Utilities from Previous Notebooks { display-mode: "form" }

def init_dense(key, n_in, n_out):
    """Initialize a single dense layer as a pytree dict."""
    w_key, b_key = jax.random.split(key)
    return {
        'w': jax.random.normal(w_key, (n_in, n_out)) * jnp.sqrt(2.0 / n_in),
        'b': jnp.zeros(n_out),
    }

def dense_forward(params, x):
    """Dense layer forward pass."""
    return x @ params['w'] + params['b']

print("Dense layer utilities ready.")
```

---

# 1. Why Python Control Flow Breaks Under JIT

Recall from Notebook 04: [`jit`](https://jax.readthedocs.io/en/latest/_autosummary/jax.jit.html) **traces** your function by running it once with abstract placeholders. Python `for` and `if` are evaluated at trace time, not at execution time. This has two consequences:

1. **`for` loops unroll**: A Python `for` loop over 100 iterations runs 100 times *at trace time*, unrolling every iteration into the compiled graph. The compiled function has no loop, just 100 copies of the loop body. This is fine for short loops but explodes graph size for large ones.

2. **`if` on a tracer raises `TracerBoolConversionError`**: A Python `if x > 0:` tries to evaluate the condition to a concrete `True`/`False` at trace time. When `x` is a tracer, JAX can't pick a branch and raises an error. The fix is `lax.cond`, which compiles **both** branches and selects one at runtime based on the actual value.

```
  Python for-loop under jit            lax.scan / lax.fori_loop under jit

     ┌─ body ─┐                          ┌─ body ─┐
     ├─ body ─┤                          │        │ ← compiled once,
     ├─ body ─┤   ← N copies in the      │        │   run N times by XLA
     ├─ body ─┤     compiled graph       └────────┘
     │  ...   │
     └─ body ─┘     graph size ∝ N         graph size = O(1)
```

```{code-cell} ipython3
# This function uses a Python for-loop
def python_loop_sum(x, n=5):
    total = 0.0
    for i in range(n):
        total = total + x
    return total

# The loop unrolls during tracing - JAX sees 5 separate add operations
print("jaxpr for python_loop_sum(x, n=5):")
print(jax.make_jaxpr(python_loop_sum)(jnp.float32(1.0)))
print("\nThe loop became 5 separate adds. For n=5 this is fine.")
print("For n=10000, you'd get a jaxpr with 10,000 operations → slow compilation.")
```

```{code-cell} ipython3
def abs_value(x):
    if x >= 0:    # Python `if` on the value of x
        return x
    else:
        return -x

# Outside jit, Python just runs the if statement normally - works fine.
print(f"Pure Python: abs_value( 3.0) = {abs_value( 3.0)}")
print(f"Pure Python: abs_value(-3.0) = {abs_value(-3.0)}")

# Under jit, x is a tracer (no actual value) and the if has nothing to compare.
try:
    jax.jit(abs_value)(jnp.float32(3.0))
except jax.errors.TracerBoolConversionError as e:
    print(f"\nUnder jit: {type(e).__name__}")
    print(f"  Short version: {str(e).splitlines()[0]}")
    print(f"  Cause: at trace time x has no value, so `if x >= 0` can't pick a branch.")
```

The solution: JAX provides **structured control flow primitives** that work under tracing. They tell JAX "there's a conditional/loop here" so it can compile both branches or the loop body.

+++

---

# 2. lax.cond and lax.switch

[`lax.cond(predicate, true_fn, false_fn, *operands)`](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.cond.html) is JAX's replacement for `if/else`. Both branches are compiled; only one executes at runtime.

> **`lax.cond` vs `jnp.where`**: `jnp.where(cond, a, b)` computes *both* `a` and `b` eagerly before selecting; the discarded branch still runs. `lax.cond` only executes the selected branch's callable at runtime. Use `lax.cond` when the branches are expensive functions or produce different shapes; use `jnp.where` for cheap element-wise selection between two pre-computed values.

```{code-cell} ipython3
def safe_abs(x):
    return lax.cond(
        x >= 0,
        lambda x: x,        # True branch
        lambda x: -x,       # False branch
        x                   # Operand passed to whichever branch runs
    )

# Works under jit!
jit_abs = jax.jit(safe_abs)
print(f"safe_abs(3.0)  = {jit_abs(jnp.float32(3.0))}")
print(f"safe_abs(-3.0) = {jit_abs(jnp.float32(-3.0))}")

# And it's differentiable
print(f"grad(safe_abs)(3.0)  = {jax.grad(safe_abs)(3.0)}")
print(f"grad(safe_abs)(-3.0) = {jax.grad(safe_abs)(-3.0)}")
```

```{code-cell} ipython3
# lax.switch(index, [fn0, fn1, fn2, ...], *operands)
# Selects and runs the function at position `index`

def activation(x, mode):
    """Apply different activations based on mode (0=relu, 1=tanh, 2=identity)."""
    return lax.switch(
        mode,
        [
            lambda x: jnp.maximum(x, 0),     # mode 0: ReLU
            lambda x: jnp.tanh(x),            # mode 1: tanh
            lambda x: x,                      # mode 2: identity
        ],
        x
    )

jit_act = jax.jit(activation)
x = jnp.array([-1.0, 0.5, 2.0])
for mode in range(3):
    print(f"Mode {mode}: {jit_act(x, mode)}")
```

> **Important**: both branches of `lax.cond` are **compiled** but only one **executes**. This means both branches must be valid code: you can't have one branch that would error. But you don't pay the runtime cost of the branch that isn't taken.

+++

---

# 3. lax.fori_loop

[`lax.fori_loop(lower, upper, body_fn, init_val)`](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.fori_loop.html) runs `body_fn` a fixed number of times, threading state through each iteration. Use it when you need a fixed-count loop but don't need to collect intermediate results.

```{code-cell} ipython3
# Compute x + x + x + ... (n times) - equivalent to x * n
def repeated_add(x, n):
    return lax.fori_loop(
        0, n,
        lambda i, total: total + x,   # body_fn(i, carry) → new_carry
        0.0                             # initial carry
    )

jit_add = jax.jit(repeated_add, static_argnames=('n',))
print(f"1.0 added 5 times: {jit_add(1.0, 5)}")
print(f"3.0 added 10 times: {jit_add(3.0, 10)}")

# Timing: fori_loop vs Python loop under jit
def python_loop_add(x, n):
    total = 0.0
    for _ in range(n):
        total = total + x
    return total

# Compare jaxpr sizes
# Note: While static_argnames is preferred for jit, jax.make_jaxpr currently only supports static_argnums
print(f"\nPython loop jaxpr (n=100): {len(str(jax.make_jaxpr(python_loop_add, static_argnums=(1,))(1.0, 100)))} chars")
print(f"fori_loop jaxpr (n=100):   {len(str(jax.make_jaxpr(repeated_add, static_argnums=(1,))(1.0, 100)))} chars")
print("\nfori_loop produces a compact jaxpr regardless of n.")
```

---

# 4. lax.scan

`scan` is the most general JAX loop. Once you understand it, the rest of the control-flow primitives are minor variations.

**The three roles in a scan:**

| Variable | Role | Analogy |
|---|---|---|
| `carry` | State that persists step-to-step | A running total in a loop |
| `xs` | The sequence being consumed, one element per step | The list you're iterating over |
| `ys` (outputs) | Values collected from each step | Appending to a results list |

`carry` flows from one step into the next. `ys` accumulates into a stacked array. After all steps, you get `(final_carry, all_ys)`. So scan is `fori_loop` with the additional ability to **collect one output per step**.

```python
def body_fn(carry, x):
    # carry: state that persists across iterations
    # x: one element from the input sequence
    new_carry = ...
    output = ...
    return new_carry, output

final_carry, all_outputs = lax.scan(body_fn, init_carry, xs)
```

Think of it as a regular `for` loop that maintains a state variable across iterations *and* keeps a list of one value per iteration, except JAX compiles the whole loop into a single device kernel instead of unrolling it.

> **Return value**: `lax.scan` returns `(final_carry, stacked_outputs)`. `final_carry` is the state *after* the last step. `stacked_outputs` is every per-step `output` stacked into a single array, equivalent to `jnp.stack([step_output_0, step_output_1, ...])`.

> **The mental model**: scan processes a sequence one element at a time, maintaining state in the carry and collecting one output per step.

+++

```text
  lax.scan(body_fn, init_carry, xs)

             xs[0]                           xs[1]
               │                               │
               ▼                               ▼
  init_carry ──► ┌─────────┐      new_carry ──►┌─────────┐
  ──────────────►│ body_fn ├──────────────────►│ body_fn ├──────► ...
                 └────┬────┘                   └────┬────┘
                      │                             │
                      ▼                             ▼
                    ys[0]                         ys[1]

  body_fn receives:  the current carry   +   one element from xs
  body_fn returns:   the new carry       +   one output to collect

  After all steps:  (final_carry, stacked([ys[0], ys[1], ...]))
```

```{code-cell} ipython3
# Cumulative sum: carry is the running total, output is the total at each step
def cumsum_step(carry, x):
    new_total = carry + x
    return new_total, new_total  # (new_carry, output)

xs = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
final_total, running_totals = lax.scan(cumsum_step, 0.0, xs)

print(f"Input:          {xs}")
print(f"Running totals: {running_totals}")
print(f"Final total:    {final_total}")
print(f"\njnp.cumsum:     {jnp.cumsum(xs)}")
print("Same result - but scan generalizes to ANY sequential computation.")
```

```{code-cell} ipython3
# A common pattern: thread a PRNG key through scan
# At each step, split the key, use one subkey, carry the other forward

def noisy_step(carry, x):
    key, running_sum = carry
    key, subkey = jax.random.split(key)
    noise = jax.random.normal(subkey) * 0.1
    new_sum = running_sum + x + noise
    return (key, new_sum), new_sum

key = jax.random.key(0)
xs = jnp.ones(10)

(final_key, final_sum), trajectory = lax.scan(noisy_step, (key, 0.0), xs)

print(f"Noisy cumulative sum (10 steps, noise_scale=0.1):")
print(f"  Trajectory: {trajectory}")
print(f"  Final sum: {final_sum:.4f} (deterministic sum would be 10.0)")
print(f"\nThe PRNG key was properly split at each step - no key reuse.")
```

```{code-cell} ipython3
# Process input through N identical layers using scan
# This is more efficient than a Python for-loop over layers

def init_stacked_layers(key, n_layers, dim):
    """Initialize N identical layers."""
    keys = jax.random.split(key, n_layers)
    # Stack parameters: each array has a leading 'layer' dimension
    return {
        'w': jax.vmap(lambda k: jax.random.normal(k, (dim, dim)) * jnp.sqrt(2.0 / dim))(keys),
        'b': jnp.zeros((n_layers, dim)),
    }

def scan_layers(stacked_params, x):
    """Process x through stacked layers using scan."""
    def layer_step(x, layer_params):
        # x is carry, layer_params is one slice of the stacked arrays
        x = x @ layer_params['w'] + layer_params['b']
        x = jnp.maximum(x, 0)
        return x, None  # carry forward, no per-step output needed

    final_x, _ = lax.scan(layer_step, x, stacked_params)
    return final_x

key = jax.random.key(0)
stacked = init_stacked_layers(key, n_layers=8, dim=32)
x = jax.random.normal(jax.random.key(1), (32,))

out = jax.jit(scan_layers)(stacked, x)
print(f"Input shape:  {x.shape}")
print(f"Output shape: {out.shape}")
print(f"Stacked weight shape: {stacked['w'].shape}  (n_layers × dim × dim)")
print(f"\nscan slices along axis 0 of each array in the pytree.")
print(f"Each step gets one layer's params - no Python loop needed.")
```

---

# 5. lax.while_loop

[`lax.while_loop(cond_fn, body_fn, init_val)`](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.while_loop.html) runs until `cond_fn` returns `False`. Use it when the number of iterations depends on the data (e.g., iterative solvers).

> **Key difference from `scan`**: Because `while_loop` doesn't know how many iterations it will run at compile time, it cannot collect outputs into a fixed-size array. The only output is the final carry: the state after the loop terminates. If you need intermediate outputs, you must write them into a pre-allocated buffer inside the carry.

```{code-cell} ipython3
# Newton's method for sqrt(a): start with a guess x, then repeatedly replace
# it with the average of x and a/x. It converges very fast - within a handful
# of iterations even for large numbers.
def newton_sqrt(a, tol=1e-6):
    """Compute sqrt(a) using Newton's iteration: x_{n+1} = (x_n + a/x_n) / 2."""

    def cond_fn(state):
        x, prev_x, _ = state
        return jnp.abs(x - prev_x) > tol

    def body_fn(state):
        x, _, n_iters = state
        new_x = (x + a / x) / 2.0
        return (new_x, x, n_iters + 1)

    # Initial guess: a/2
    init_state = (a / 2.0, 0.0, 0)
    final_x, _, n_iters = lax.while_loop(cond_fn, body_fn, init_state)
    return final_x, n_iters

# Test
jit_sqrt = jax.jit(newton_sqrt)
for val in [2.0, 9.0, 100.0, 12345.0]:
    result, iters = jit_sqrt(val)
    exact = jnp.sqrt(val)
    print(f"sqrt({val:>8.1f}) = {result:.8f}  (exact: {exact:.8f}, iters: {iters})")

print("\nwhile_loop terminates when convergence is reached - iteration count varies.")
```

---

# 6. Decision Table

| Scenario | Use | Why |
|---|---|---|
| Fixed iterations, need intermediates | **`lax.scan`** | Collects outputs, efficient under JIT |
| Fixed iterations, only final result | **`lax.fori_loop`** | Slightly simpler API than scan |
| Small fixed loop (< ~10 iters) | **Python `for`** | Unrolling is fine for small loops |
| Data-dependent termination | **`lax.while_loop`** | Only option for "run until converged" |
| Conditional (if/else) | **`lax.cond`** | Traced conditional, both branches compiled |
| Multi-way conditional | **`lax.switch`** | Selects one of N branches by index |
| Simple element-wise conditional | **[`jnp.where`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.where.html)** | No branch overhead, evaluates both sides |

The most common is **`scan`**: it handles training loops, RNNs, sequential models, autoregressive generation, and any "process a sequence while maintaining state" pattern.

+++

---

# 7. Example: Simulating a Random Walk

Instead of complex neural networks, let's use `scan` to simulate a cumulative random walk, which is mathematically just a cumulative sum over randomly generated steps. This shows how state builds up over time.

```{code-cell} ipython3
def advance_walk(current_position, step_value):
    new_position = current_position + step_value
    return new_position, new_position

key = jax.random.key(0)
steps = jax.random.normal(key, shape=(10,))
initial_pos = 0.0

final_pos, history = jax.lax.scan(advance_walk, initial_pos, steps)
print("Steps:", steps)
print("History:", history)
print("Final Position:", final_pos)
```

---

# 8. Summary

## Key Takeaways

- **Python loops unroll at trace time**: a Python `for` loop inside a JIT function compiles into a massive sequence of operations. Good for small static loops, terrible for large ones.

- **`lax.scan` is the main loop primitive**: use it for recurrent computations that carry state across iterations. It compiles to an efficient, bounded XLA loop regardless of sequence length.

- **`lax.cond` handles dynamic branching**: it replaces Python `if/else` under JIT. Both branches must return the same shapes and dtypes.

- **`lax.while_loop` handles dynamic termination**: use it when the number of iterations depends on the data (e.g., convergence checks).

## What's Next

In **Notebook 07: Training From Scratch**, we'll put all the JAX fundamentals (pytrees, `grad`, `jit`, and now structured control flow) together to write a complete training loop from raw JAX, with no frameworks. That's the baseline that motivates Flax and Optax in the notebooks that follow.

+++

---

# 9. Exercises

1. **lax.cond**: Write a JIT-compiled function `clamp(x, lo, hi)` that returns `lo` if `x < lo`, `hi` if `x > hi`, and `x` otherwise. Use `lax.cond` (you may need to nest two calls). Test with several values.

2. **lax.scan**: Use `lax.scan` to compute a running product (cumulative product) of a 1D array. The carry should hold the running product, and the output should be the product at each step. Compare your result with `jnp.cumprod`.

3. **lax.while_loop**: Use `lax.while_loop` to start with `x = 100.0` and repeatedly divide it by 2 until it is less than 1.0. Count how many steps it takes.
