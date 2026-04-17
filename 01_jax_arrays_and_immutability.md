---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: .venv
  language: python
  name: python3
---

# Part 1: JAX Arrays and Immutability

> **Prerequisites**: Basic Python knowledge (variables, functions, loops).

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

---

# 1. Introduction to JAX

## 1.1 The Core Idea

JAX is a Python library for numerical computing on **n-dimensional arrays** (defined in Section 2). It has two halves:

1. **An array library, [`jax.numpy`](https://jax.readthedocs.io/en/latest/jax.numpy.html)**: A NumPy-compatible interface for high-performance array operations that can execute on a CPU, GPU (Graphics Processing Unit), or TPU (Tensor Processing Unit). GPUs and TPUs are hardware accelerators optimized for parallel array computations, running them much faster than a standard CPU.

2. **A small set of *function transformations*** in the top-level [`jax`](https://jax.readthedocs.io/en/latest/jax.html) module. Each one takes a Python function and gives you back a new, transformed function:

   * [`jit`](https://jax.readthedocs.io/en/latest/_autosummary/jax.jit.html): returns a compiled version of the function for faster execution on device.

   * [`grad`](https://jax.readthedocs.io/en/latest/_autosummary/jax.grad.html): returns a function that computes the gradient (derivative) of a function (see Notebook 03).

   * [`vmap`](https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html): returns a vectorized version of the function that runs over a batch of inputs at once, in parallel.

We'll cover all three in later notebooks. Don't worry if the names mean nothing yet.

## 1.2 The Power of Composition

Because each transformation takes a function and returns a function, you can stack them. Starting with a function `f` that processes one input, you can wrap it in `grad` to get its gradient, wrap *that* in `vmap` to run over a batch, then wrap *that* in `jit` to compile the whole thing. The diagram below shows the layering. You won't write this for real until later notebooks; it's here so you've seen the shape.

```
    jit ( vmap ( grad ( f ) ) )
     │      │      │    │
     │      │      │    └──- (1) your pure function
     │      │      └──────── (2) → function returning the gradient
     │      └─────────────── (3) → function batched over inputs
     └────────────────────── (4) → function compiled to device code

    Each transformation takes a function and returns a new function.
    Composition just nests them.
```

## 1.3 A Shift in Programming Paradigm

Most Python code is written **imperatively**: you issue a sequence of commands, variables change as the program runs, and the order of those mutations matters. JAX works differently.

* **Pure functions**: A JAX function must have **no side effects**: it cannot mutate global variables, modify inputs in-place, or depend on any state other than its arguments. Its output depends *only* on its inputs.

* **Explicit state**: Data and parameters are passed in as arguments and returned as values. Nothing is stored or modified "inside" the function between calls.

By adhering to these constraints, JAX guarantees that its transformations (`jit`, `grad`, `vmap`) are safe to apply to any function you write. You'll see exactly why in Section 3.

+++

---

# 2. JAX Arrays

## 2.1 What is an Array?

If you've used a Python list of numbers, you already have the basic intuition: an **array** is a grid of numbers (an n-dimensional structure). The differences from a list are what make arrays useful for math:

- **Every element has the same type** (e.g., `float32`, `int32`). This element type is called the **dtype**.

- **The shape is fixed**: a tuple of axis lengths. A `(3, 4)` array has 3 rows and 4 columns. A `(8,)` array is a 1-D vector of length 8. A `()` array is a single number with no axes, called a **scalar**.

- **Mathematical operations are applied element-wise** and executed in parallel on accelerators without Python loops.

A few terms you'll see throughout the series:

| Term | Meaning |
|---|---|
| **dtype** | Element type (`float32`, `int32`, `bool`, etc.) |
| **shape** | Tuple of axis lengths. `(3, 4)` means 3 rows × 4 columns |
| **axis** | One dimension of the array. A `(3, 4)` 2D array has two axes: axis 0 (rows) and axis 1 (columns) |
| **rank** | Number of axes. A scalar is rank 0, a vector is rank 1, a matrix is rank 2 |
| **scalar** | A single number, in JAX scalars have the shape `()` |

## 2.2 Key Properties

**NumPy** is a popular Python library for numerical computing. JAX's array API was designed to be very similar to NumPy's. If you've used NumPy, the left column will look familiar. If you haven't, no worries.

| | **NumPy** | **JAX** |
|---|---|---|
| **Paradigm** | Imperative | Functional transformations |
| **Mutability** | Mutable arrays | Immutable arrays |
| **Compilation** | None (interpreted) | [`jit`](https://jax.readthedocs.io/en/latest/_autosummary/jax.jit.html) traces → compiles to device code |
| **Differentiation** | None | [`grad(f)`](https://jax.readthedocs.io/en/latest/_autosummary/jax.grad.html) returns a gradient function |
| **Batching** | Manual vectorization/reshaping | [`vmap(f)`](https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html) returns a vectorized function |


In JAX, transformations return new functions. [`grad(f)`](https://jax.readthedocs.io/en/latest/_autosummary/jax.grad.html) doesn't mutate `f` or compute a gradient. It returns a *new function* that computes the gradient (representing the derivative or slope of the function; covered properly in Notebook 03). 

## 2.3 The jax.numpy API

JAX provides `jax.numpy` (commonly imported as `jnp`), a comprehensive API for array math. You create arrays, perform arithmetic, slice, and reshape them using familiar Python syntax. The next two cells show common examples.

```{code-cell} ipython3
# Three arrays of different shapes
scalar = jnp.array(42.0)              # shape () - a single number
vector = jnp.array([1.0, 2.0, 3.0])   # shape (3,) - 1-D, three elements
matrix = jnp.zeros((2, 3))            # shape (2, 3) - 2 rows, 3 columns

for name, a in [("scalar", scalar), ("vector", vector), ("matrix", matrix)]:
    print(f"  {name}: shape={a.shape}, dtype={a.dtype}, rank={a.ndim}")
```

```{code-cell} ipython3
# Array creation
a = jnp.array([1.0, 2.0, 3.0])
b = jnp.ones((3, 3))
c = jnp.linspace(0, 1, 5)

print("Array creation:")
print(f"  a = {a}")
print(f"  b.shape = {b.shape}")
print(f"  c = {c}")

# Standard operations - same API
print(f"\nOperations:")
print(f"  jnp.dot(a, a) = {jnp.dot(a, a)}")
print(f"  jnp.mean(b)   = {jnp.mean(b)}")
print(f"  jnp.sin(c)    = {jnp.sin(c)}")

# Slicing, reshaping, broadcasting - all familiar
x = jnp.arange(12).reshape(3, 4)
print(f"\nSlicing and reshaping:")
print(f"  x =\n{x}")
print(f"  x[0, :2] = {x[0, :2]}")
print(f"  x.T shape = {x.T.shape}")
```

---

# 3. Immutability and Pure Functions

Here is one of the big differences between JAX and ordinary Python/NumPy. In NumPy you can modify an array element in place:

```python
a = np.zeros(5)
a[2] = 99.0  # In-place mutation
```

JAX arrays are **immutable**. The equivalent operation in JAX uses a different syntax:

```{code-cell} ipython3
# --- NumPy: in-place mutation ---
np_arr = np.zeros(5)
print(f"NumPy initial:  {np_arr}")
np_arr[2] = 99.0
print(f"NumPy (mutated in place):  {np_arr}")

# --- JAX: .at[].set() returns a NEW array ---
jax_arr = jnp.zeros(5)
print(f"\nJAX initial: {jax_arr}")
jax_new = jax_arr.at[2].set(99.0)

print(f"JAX original (not mutated): {jax_arr}")
print(f"JAX new (copy with changed data):   {jax_new}")
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

JAX's [`.at[]`](https://jax.readthedocs.io/en/latest/_autosummary/jax.Array.at.html) API supports all the update operations you'd expect:

| NumPy (in-place) | JAX (returns new array) |
|---|---|
| `a[idx] = val` | [`a.at[idx].set(val)`](https://jax.readthedocs.io/en/latest/_autosummary/jax.Array.at.html) |
| `a[idx] += val` | [`a.at[idx].add(val)`](https://jax.readthedocs.io/en/latest/_autosummary/jax.Array.at.html) |
| `a[idx] *= val` | [`a.at[idx].mul(val)`](https://jax.readthedocs.io/en/latest/_autosummary/jax.Array.at.html) |
| `a[idx] = np.maximum(a[idx], val)` | [`a.at[idx].max(val)`](https://jax.readthedocs.io/en/latest/_autosummary/jax.Array.at.html) |
| `a[idx] = np.minimum(a[idx], val)` | [`a.at[idx].min(val)`](https://jax.readthedocs.io/en/latest/_autosummary/jax.Array.at.html) |

> **Note on `.max` and `.min`**: `a.at[idx].max(val)` keeps whichever of the two values is larger: the existing element at `idx` or `val`. It is an element-wise conditional update and does **not** involve the global min/max of an array.

```{code-cell} ipython3
# Various matrix updates
x = jnp.zeros((4, 4))
print("Original matrix:")
print(x)

x = x.at[0, :].set(1.0)          # Set first row
print("\nSet first row result:")
print(x)

x = x.at[:, -1].add(10.0)        # Add to last column
print("\nAdd to last column result:")
print(x)

x = x.at[jnp.diag_indices(4)].set(-1.0)  # Set diagonal
print("\nSet diagonal result:")
print(x)
```

## 3.1 Why Immutability? The Functional Contract

> **JAX transformations only see what flows through function arguments and return values. Everything else is invisible.**

Immutability is **not** just a stylistic preference. It's fundamental to how JAX's transformations work. When you apply [`jit`](https://jax.readthedocs.io/en/latest/_autosummary/jax.jit.html), [`grad`](https://jax.readthedocs.io/en/latest/_autosummary/jax.grad.html), or [`vmap`](https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html) to a function, JAX **traces** it. Tracing is the process where JAX runs your function once with special placeholder values (called *tracers*) instead of real numbers. Instead of executing concrete calculations on numbers, JAX tracks the flow of these tracers to record the sequence of operations, building a static computational graph (called a jaxpr) that can be optimized and compiled. Anything that doesn't flow through the arguments and return value (global state, in-place mutation, print statements, random state) is invisible to the trace.

```text
                  ┌─────────────────────────────┐
                  │ Apply JAX transformation    │
                  └──────────────┬──────────────┘
                                 │
                  ┌──────────────▼──────────────┐
                  │ Trace function with         │
                  │ abstract tracers            │
                  └──────────────┬──────────────┘
                                 │
              ┌──────────────────┴──────────────────┐
              │                                     │
┌─────────────▼─────────────┐         ┌─────────────▼─────────────┐
│ Pure Array Operations     │         │ Python Side Effects       │
│ (jnp.add, jnp.dot, etc.)  │         │ (print, append, mutate)   │
└─────────────┬─────────────┘         └─────────────┬─────────────┘
              │                                     │
              ▼                                     ▼
┌───────────────────────────┐         ┌───────────────────────────┐
│ Recorded in jaxpr         │         │ Ignored! Invisible to JAX │
└───────────────────────────┘         └───────────────────────────┘
```

A **jaxpr** is JAX's intermediate representation: the recorded list of operations your function performed during tracing.

+++

## 3.2 Why This Matters

The functional contract (pure functions in, transformed functions out) enables **composable transformations**. Because [`grad`](https://jax.readthedocs.io/en/latest/_autosummary/jax.grad.html) can trust that a function has no hidden state, it can safely transform it. Because [`jit`](https://jax.readthedocs.io/en/latest/_autosummary/jax.jit.html) can trust that a function is deterministic given its inputs, it can safely cache the compiled version and re-use it.

This is what makes `jax.grad`, `jax.jit`, and `jax.vmap` safe to apply to any function: they rely on being able to trace it without worrying about hidden state changing underneath.

If you find yourself fighting immutability, you're likely trying to write imperative code in a functional system. The fix is almost always to pass state explicitly through function arguments and return values.

*(Note: Notebook 04 covers tracing, `jaxpr`, and JIT compilation in more depth.)*

+++

---

# 4. Summary

## Key Takeaways

- **JAX arrays are immutable**: you cannot mutate elements in-place. Use `x = x.at[i].set(val)` to return a modified copy.

- **Functions must be pure**: JAX transformations require functions to have no side effects, such as mutating global state or relying on implicit inputs.

## What's Next

In **Notebook 02: Broadcasting and Randomness**, we'll cover how operations combine arrays of different shapes (broadcasting), and why JAX threads random numbers through explicit keys instead of hidden global state, which is a direct consequence of the pure-function rule from this notebook.

+++

---

# 5. Exercises

1. **Array operations**: Create a `(5, 5)` array of ones using `jnp.ones`. Use `.at[].set()` to put zeros along the diagonal. Verify the result has 1s everywhere except the diagonal.

2. **Immutability check**: Create an array `a = jnp.arange(10)`. Apply three chained `.at[].set()` operations to change indices 0, 5, and 9. Verify that `a` is unchanged after all three operations.

3. **Filtering values**: Create a 1D array of numbers from -5 to 5 (try `jnp.arange(-5, 6)`). Use `jnp.maximum` with `0` to create a new array where all negative numbers are replaced by 0. Verify the original array is unchanged.
