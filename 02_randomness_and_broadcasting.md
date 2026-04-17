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

# Part 2: Broadcasting and Randomness

```{code-cell} ipython3
# @title Setup { display-mode: "form" }

import jax
import jax.numpy as jnp
import numpy as np

print(f"JAX version:     {jax.__version__}")
print(f"Devices:         {jax.devices()}")
print(f"Default backend: {jax.default_backend()}")
```

# 1. Broadcasting

**Broadcasting** is the rule that lets you do math between arrays of different shapes, such as adding a `(4,)` vector to a `(4, 3)` matrix. Instead of writing a loop, you just write `matrix + vector` and JAX figures out how to make the shapes line up.

There's just one rule to remember: **JAX (and NumPy) line up shapes from the right**. Two shapes are compatible at a given dimension if they're equal, or if one of them is `1`.

```
  Lining up from the right:

     matrix  (4, 3)             matrix  (4, 3)
     vector  (   4,)            column  (4, 1)
                ^^                          ^^
     trailing 3 vs 4            trailing 3 vs 1
     MISMATCH - error           OK - the 1 stretches to 3
```

So a bare `(4,)` vector lines up against the *columns* of a `(4, 3)` matrix (not the rows), which fails because 4 ≠ 3. To add the vector to each *row*, you have to make it a `(4, 1)` column first, and the standard way to do that is `vector[:, None]` (you can also use [`.reshape()`](https://jax.readthedocs.io/en/latest/_autosummary/jax.Array.reshape.html) or `jnp.newaxis`, which is just another name for `None` in this context).

```{code-cell} ipython3
vector = jnp.array([1.0, 2.0, 3.0, 4.0])    # Shape: (4,)
matrix = jnp.ones((4, 3))                   # Shape: (4, 3)

print(f"vector shape: {vector.shape}")
print(f"vector: {vector}\n")
print(f"matrix shape: {matrix.shape}")
print(f"matrix: {matrix}\n")

# Broadcasting aligns shapes from the RIGHT (trailing axes).
# vector (4,) is treated as (1, 4) -> trailing axes are 4 vs 3: MISMATCH!
# Even though the matrix has 4 rows and the vector has 4 elements,
# JAX aligns against columns, not rows.
try:
    res = matrix + vector
except Exception as e:
    print(f"Error when performing (matrix + vector): {e}\n")

# Fix: inject a dummy axis to make it a column vector (4, 1).
# Now trailing axes are 1 vs 3: broadcasts freely to (4, 3).
column = vector[:, None]                    # Shape: (4, 1)
print("Add a dummy axis: column = vector[:, None]")
print(f"column shape : {column.shape}\n")

print(f"Matrix: {matrix}")
print(f"Column: {column}\n")

result = matrix + column                       # (4, 3) + (4, 1) -> (4, 3)
print(f"Matrix + Column Success! Result shape: {result.shape}")
print(f"Result: {result}")
```

## 1.1 Common JAX Idioms

The table below covers the most common JAX patterns. The left column shows the NumPy equivalent. If you've used NumPy, you'll recognize the operation; if not, focus on the **JAX Equivalent** and **Why** columns.

| NumPy Pattern | JAX Equivalent | Why |
|---|---|---|
| `a[i] = v` | [`a = a.at[i].set(v)`](https://jax.readthedocs.io/en/latest/_autosummary/jax.Array.at.html) | Immutability |
| `np.random.randn(3)` | [`jax.random.normal(key, (3,))`](https://jax.readthedocs.io/en/latest/_autosummary/jax.random.normal.html) | Explicit PRNG (Section 2) |
| `a.sort()` | [`a = jnp.sort(a)`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.sort.html) | No in-place sort due to immutability|
| `np.concatenate` | [`jnp.concatenate`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.concatenate.html) | Same API, different backend |
| `if a > 0:` (on array) | [`jnp.where(a > 0, ...)`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.where.html) | Traceable control flow |
| `for i in range(n):` | Often [`jax.lax.scan`](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html) | Efficient under JIT ([Notebook 06](06_control_flow.ipynb)) |

+++

---

# 2. Random Number Generation

## 2.1 Why Randomness Matters

Numerical computing uses randomness in several common places:

- **Setting up a neural network's weights** before training. If every weight started at the same value, every neuron in a layer would do the same thing. You need different random starting points so each neuron can learn something different.

- **Shuffling data** between training passes so the model doesn't memorize the order.

- **Adding noise** to inputs or to the model itself as part of certain training tricks.

- **Simulating random processes** (a coin flip, a random walk, a Monte Carlo simulation).

In ordinary Python and NumPy, randomness is easy: `np.random.randn(3)` just gives you a number. There's a **PRNG** (pseudo-random number generator) hidden inside the library, and each call quietly advances its state:

```python
np.random.seed(42)
x = np.random.randn(3)  # draws from the hidden state, then advances it
y = np.random.randn(3)  # different numbers (because the state moved on)
```

JAX can't do this. That hidden state is a side effect, and side effects are invisible to JAX's transformations (Notebook 01). Instead, JAX makes the PRNG state explicit: you hold it in a variable called a **key** and pass it into every random function.

> **`jax.jit` shows up early here.** We cover it properly in Notebook 04; for now all you need to know is that it *compiles* a function by **tracing** it once (Notebook 01) and reusing the result on later calls. The example below shows what goes wrong when that trace-once step captures a non-JAX random call: the random value gets frozen into the compiled function.

```{code-cell} ipython3
# np.random inside a jitted function gives the SAME result every call
# because the random call happens at trace time, not execution time.

# The @ decorator is Python shorthand to pass this function through jax.jit()
@jax.jit
def bad_random(x):
    noise = np.random.randn()  # Uses NumPy global state - BAD
    return x + noise

# Call it multiple times
results = [float(bad_random(jnp.array(0.0))) for _ in range(5)]
print("np.random inside jit:")
for i, r in enumerate(results):
    print(f"  Call {i+1}: {r:.6f}")
print("\nSame result every time! The np.random.randn() was baked into the")
print("compiled trace as a constant.")
```

## 2.2 Keys, Splitting, and Subkeys

JAX replaces global state with **explicit PRNG keys**. A key is a pair of 32-bit integers that deterministically produces random numbers.

The mental model:

> **A key acts as the PRNG state. Splitting generates new states.**

You start with one key. Every time you need randomness, you **split** the key into subkeys (one for each random draw), and you never reuse a key.

```text
┌─────────────────┐
│ key=jax.random.key(42) │
└────────┬────────┘
         │
         ▼ split()
  ┌──────┴──────┐
  │             │
┌─▼─┐       ┌───▼────┐
│key│       │ subkey ├──► random.normal(...)  ← draw 1
└─┬─┘       └────────┘
  │
  ▼ split()
┌─┴─┐       ┌────────┐
│key│       │ subkey ├──► random.normal(...)  ← draw 2
└─┬─┘       └────────┘
  │
  ▼ split()
┌─┴─┐       ┌────────┐
│key│       │ subkey ├──► random.normal(...)  ← draw 3
└───┘       └────────┘
```

Each split consumes the current key and produces a fresh key (for the next split) plus one or more subkeys (to feed into `random.normal`, `random.uniform`, etc.). A subkey is never reused; the key is never reused.

```{code-cell} ipython3
# Create a key from a seed
key = jax.random.key(42)
print(f"Original key: {key}")
print(f"Key dtype: {key.dtype}, shape: {key.shape}")

# Split into subkeys
key, subkey1, subkey2 = jax.random.split(key, 3)
print(f"\nAfter split:")
print(f"  New key:  {key}")
print(f"  Subkey 1: {subkey1}")
print(f"  Subkey 2: {subkey2}")

# Draw random numbers using subkeys
x = jax.random.normal(subkey1, shape=(3,))
y = jax.random.normal(subkey2, shape=(3,))
print(f"\nDrawn from subkey1: {x}")
print(f"Drawn from subkey2: {y}")
print("\nDifferent subkeys → different (independent) random numbers.")
```

> **What's `key<fry>`?** That's the internal name of JAX's random-key dtype: the underlying integers are arranged according to a hash function called "fry". You can treat a key as an opaque blob; the only operations you need are `jax.random.split` (to make new keys) and passing it into a sampling function like `jax.random.normal`.

```{code-cell} ipython3
key = jax.random.key(0)

# Reusing the same key gives the SAME random numbers
a = jax.random.normal(key, shape=(3,))
b = jax.random.normal(key, shape=(3,))
print(f"Same key, call 1: {a}")
print(f"Same key, call 2: {b}")
print(f"Identical? {jnp.allclose(a, b)}")
print("\nThis is intentional: same key in, same numbers out. That's what reproducibility means here.")
print("If you want different numbers, use different keys (split first).")
```

## 2.3 The Splitting Pattern

The standard pattern for managing keys is to split before each usage:

```python
key = jax.random.key(seed)

# Every time you need randomness:
key, subkey = jax.random.split(key)
samples = jax.random.normal(subkey, shape=(...))
```

Split first, draw second. The `key` variable advances each time, and `subkey` is consumed for the draw. The key is never reused; the subkey is never reused.

> **Note**: In newer JAX versions (0.4.26+), `jax.random.key(seed)` is the preferred way to create keys, replacing the legacy `jax.random.PRNGKey(seed)`. Both work identically, but `key()` is the modern API.

```{code-cell} ipython3
key = jax.random.key(0)

print("Sequential draws with proper splitting:")
for i in range(5):
    key, subkey = jax.random.split(key)
    sample = jax.random.normal(subkey, shape=(1,))
    print(f"  Draw {i+1}: {sample.item():.6f}")

print("\nEach draw is different because each uses a different subkey.")
print("But the ENTIRE sequence is reproducible - run this cell again and you get the same numbers.")
```

## 2.4 PRNG and Transformations

The explicit key system isn't just for reproducibility. It's what makes randomness work correctly under [`jit`](https://jax.readthedocs.io/en/latest/_autosummary/jax.jit.html) and [`vmap`](https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html).

| Pattern | How It Works |
|---|---|
| **jit** | Key flows through args → no side effects → safe to compile |
| **vmap** | Split N keys, vmap over them → independent randomness per batch element |
| **grad** | Key is non-differentiable → passes through unchanged |

```{code-cell} ipython3
# Works perfectly under jit - key is an explicit argument
@jax.jit
def random_add(key, x):
    noise = jax.random.normal(key, shape=x.shape)
    return x + 0.1 * noise

key = jax.random.key(42)
x = jnp.ones(4)

# Each call with a different key gives different results
key, k1, k2 = jax.random.split(key, 3)
print("jit + PRNG:")
print(f"  Key 1: {random_add(k1, x)}")
print(f"  Key 2: {random_add(k2, x)}")

# Works perfectly under vmap - one key per batch element.
# (vmap is covered properly in Notebook 04. For now, think of jax.vmap(f)(xs)
#  as "run f once per row of xs, in parallel, and stack the results.")
keys = jax.random.split(key, 8)  # 8 independent keys
batch = jnp.ones((8, 4))
results = jax.vmap(random_add)(keys, batch)
print(f"\nvmap + PRNG (8 independent samples):")
print(f"  Shape: {results.shape}")
print(f"  Row 0: {results[0]}")
print(f"  Row 1: {results[1]}")
print("  Each row used a different key → independent noise.")
```

---

# 3. Summary

## Key Takeaways

- **Broadcasting is automatic shape alignment**: operations on arrays of different shapes work as long as the trailing dimensions match (or are 1).

- **`x[:, None]` adds a length-1 axis**: the standard tool for turning a vector into a column so it broadcasts the way you want.

- **Randomness is explicit**: pass a PRNG key into every random function. There is no global state.

- **Same key, same numbers**: JAX guarantees reproducibility by giving you the same draw whenever you pass the same key. To get fresh draws, call `jax.random.split` first.

## What's Next

In **Notebook 03: Autodiff**, we'll meet `jax.grad`, the first of JAX's three core transformations. Because your code is now pure (Notebook 01) and shape-aligned (this notebook), autodiff "just works" through arbitrary Python.

+++

---

# 4. Exercises

1. **Broadcasting basics**: Create a 1D array `v = jnp.array([1, 2, 3])`. Add a scalar `5` to it. Then create a 2D matrix of shape `(3, 3)` filled with ones and add `v` to it.

2. **Key splitting practice**: Generate 5 independent random normal samples (scalars) using a single starting key `jax.random.key(0)`. Print each one. Then re-run with the same starting key and verify the results are identical.

3. **PRNG under jit**: Write a `@jax.jit` function that takes a key and returns a random normal sample. Call it with two different keys. Do you get different results? Now call it twice with the *same* key. What do you get?
