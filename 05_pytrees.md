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

# Part 5: Pytrees and Model Parameters

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
# @title MLP Utilities from Previous Notebooks { display-mode: "form" }

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
    """Forward pass for a SINGLE input vector."""
    for i, (w, b) in enumerate(params):
        x = x @ w + b
        if i < len(params) - 1:
            x = jnp.maximum(x, 0)
    return x

key = jax.random.key(42)
params = init_mlp_params(key, [8, 64, 32, 1])
print("MLP reconstructed: 8 → 64 → 32 → 1")
```

---

# 1. Pytrees: JAX's Universal Container

So far, our model parameters have been a list of `(weight, bias)` tuples, and we updated them with a list comprehension that destructures the tuples by hand. That works for one or two layers; it doesn't scale to dozens. JAX has a better abstraction: **pytrees**.

> **A pytree is any nested structure of Python containers (dicts, lists, tuples) with array leaves.**

JAX transforms understand pytrees natively. When you pass a pytree to [`grad`](https://jax.readthedocs.io/en/latest/_autosummary/jax.grad.html), you get back a pytree with the same structure. When you pass a pytree to [`jit`](https://jax.readthedocs.io/en/latest/_autosummary/jax.jit.html), every leaf gets traced.

```text
  params = {                          grads = {
      'layer0': {                         'layer0': {
          'w': array(...),       grad →       'w': array(...),
          'b': array(...),       ─────►       'b': array(...),
      },                                  },
      'layer1': { ... },                  'layer1': { ... },
  }                                   }
```

> **What `jax.tree.map(f, tree_a, tree_b)` does**: it walks both pytrees in parallel (they must have the same structure), applies `f(leaf_a, leaf_b)` at every matching pair of leaves, and returns a new pytree with the same structure. Think of it as `zip` for nested structures.

<br>

```text
  params (dict)                        grads (dict - same structure)
  ┌─────────────────────┐              ┌─────────────────────┐
  │ 'layer0': {         │    grad()    │ 'layer0': {         │
  │   'w': f32[8, 64]   ├─────────────►│   'w': f32[8, 64]   │
  │   'b': f32[64]      │              │   'b': f32[64]      │
  │ }                   │              │ }                   │
  │ 'layer1': {         │              │ 'layer1': {         │
  │   'w': f32[64, 32]  │              │   'w': f32[64, 32]  │
  │   'b': f32[32]      │              │   'b': f32[32]      │
  │ }                   │              │ }                   │
  └─────────────────────┘              └─────────────────────┘

  Leaves: the arrays at the bottom of the tree.
  Nodes: the Python dicts/lists/tuples that contain them.
```

```{code-cell} ipython3
# "Pytree" is just JAX's term for a nested structure of Python containers with array leaves.
# We can inspect a pytree's hierarchy using jax.tree.

print("1. The shape of each parameter array:\n")
print(jax.tree.map(lambda arr: arr.shape, params))

print("\n2. The structure (treedef) JAX sees:\n")
print(jax.tree.structure(params))

print("\nA pytree has 'nodes' (the Python lists/tuples/dicts) and 'leaves' (the JAX arrays).")
print("JAX transforms can map operations across leaves automatically.")
```

```{code-cell} ipython3
# Any nested structure of dicts/lists/tuples with array leaves is a pytree
example_pytree = {
    'weights': jnp.ones((3, 2)),
    'biases': jnp.zeros(2),
    'config': {
        'scale': jnp.array(0.1),
    }
}

# jax.tree has utilities for working with pytrees
leaves = jax.tree.leaves(example_pytree)
print(f"Number of leaves: {len(leaves)}")
for i, leaf in enumerate(leaves):
    print(f"  Leaf {i}: shape {leaf.shape}, dtype {leaf.dtype}")

# tree structure
structure = jax.tree.structure(example_pytree)
print(f"\nStructure: {structure}")
```

```{code-cell} ipython3
# tree_map applies a function to every leaf, preserving structure
doubled = jax.tree.map(lambda x: x * 2, example_pytree)

print("Original:")
print(f"  weights[0,:] = {example_pytree['weights'][0]}")
print(f"  scale = {example_pytree['config']['scale']}")

print("\nDoubled:")
print(f"  weights[0,:] = {doubled['weights'][0]}")
print(f"  scale = {doubled['config']['scale']}")

# tree_map with multiple trees (must have matching structure)
a = {'x': jnp.array([1.0, 2.0]), 'y': jnp.array(3.0)}
b = {'x': jnp.array([10.0, 20.0]), 'y': jnp.array(30.0)}

added = jax.tree.map(lambda a, b: a + b, a, b)
print(f"\nAdding two pytrees: {added}")
```

## 1.1 Restructuring Our MLP as a Pytree

Our list-of-tuples parameter store *is* already a pytree (lists and tuples count as pytree nodes). But a list of dicts is more readable and makes the leaf names explicit, so we'll switch to that for the rest of the notebook.

```{code-cell} ipython3
def init_mlp_pytree(key, layer_sizes):
    """Initialize MLP as a list of {'w': ..., 'b': ...} dicts."""
    params = []
    for i in range(len(layer_sizes) - 1):
        key, w_key = jax.random.split(key)
        n_in, n_out = layer_sizes[i], layer_sizes[i + 1]
        params.append({
            'w': jax.random.normal(w_key, (n_in, n_out)) * jnp.sqrt(2.0 / n_in),
            'b': jnp.zeros(n_out),
        })
    return params

def mlp_forward_pytree(params, x):
    """Forward pass using pytree params."""
    for i, layer in enumerate(params):
        x = x @ layer['w'] + layer['b']
        if i < len(params) - 1:
            x = jnp.maximum(x, 0)
    return x

key = jax.random.key(42)
params_pt = init_mlp_pytree(key, [8, 64, 32, 1])

print("Pytree parameter structure:")
for i, layer in enumerate(params_pt):
    print(f"  Layer {i}: w {layer['w'].shape}, b {layer['b'].shape}")

# Verify: count total parameters
n_params = sum(x.size for x in jax.tree.leaves(params_pt))
print(f"\nTotal parameters: {n_params:,}")
```

---

# 2. Pytrees and Transforms

The real power of pytrees: JAX transforms understand them natively. [`grad`](https://jax.readthedocs.io/en/latest/_autosummary/jax.grad.html) through a pytree returns a matching pytree of gradients. `jax.tree.map` over params and grads gives you a clean update rule.

```{code-cell} ipython3
def loss_fn(params, x, y):
    pred = mlp_forward_pytree(params, x)
    return jnp.mean((pred - y) ** 2)

# grad w.r.t. params (first arg) - returns a pytree matching params
x_sample = jax.random.normal(jax.random.key(0), (8,))
y_sample = jnp.array(1.0)

grads = jax.grad(loss_fn)(params_pt, x_sample, y_sample)

print("Gradient structure (matches params exactly):")
for i, layer in enumerate(grads):
    print(f"  Layer {i}: dw {layer['w'].shape}, db {layer['b'].shape}")
```

```{code-cell} ipython3
# SGD update in one line using tree_map
lr = 0.01
updated_params = jax.tree.map(lambda p, g: p - lr * g, params_pt, grads)

# Verify the update happened
print("Before update:")
print(f"  Layer 0 w mean: {jnp.mean(params_pt[0]['w']):.6f}")
print(f"\nAfter update:")
print(f"  Layer 0 w mean: {jnp.mean(updated_params[0]['w']):.6f}")
print("\nOne `tree.map` line walks both pytrees in parallel and applies the SGD update to every leaf - no manual destructuring of tuples per layer.")
```

---

# 3. Advanced Pytrees

An advanced pattern worth knowing: masking the tree so updates apply only to some parameters (for example, freezing certain layers during training).

```
  Freeze mask: same tree structure, boolean leaves

      params               grads                mask                   after update
     ┌───────────┐        ┌───────────┐        ┌───────────┐          ┌───────────┐
     │ Layer 0   │        │ dw, db    │        │ True      │ frozen   │ unchanged │
     │   w, b    │   +    │           │   +    │  True     │    =     │           │
     ├───────────┤        ├───────────┤        ├───────────┤          ├───────────┤
     │ Layer 1   │        │ dw, db    │        │ False     │ trains   │ p - lr·g  │
     │   w, b    │        │           │        │  False    │          │           │
     └───────────┘        └───────────┘        └───────────┘          └───────────┘

     jax.tree.map(lambda p, g, m: p if m else p - lr*g,
                  params, grads, mask)
```

+++

## 3.1 Filtering: Selective Parameter Updates

`jax.tree.map` applies the same function to every leaf. But sometimes you want to treat different parameters differently, for example freezing some layers during training.

The trick: walk the params, grads, and a boolean mask in parallel with `jax.tree.map`. For frozen leaves, return the param unchanged. For trainable leaves, apply the usual `param - lr * grad`. The frozen layer never receives an update.

```{code-cell} ipython3
# Strategy: create a "freeze mask" pytree with the same structure.
# True = freeze (keep param unchanged), False = train normally.

def make_freeze_mask(params, frozen_layers):
    """Create a boolean mask pytree. True = frozen."""
    mask = []
    for i, layer in enumerate(params):
        frozen = i in frozen_layers
        mask.append(jax.tree.map(lambda _: frozen, layer))
    return mask

# Freeze layer 0
mask = make_freeze_mask(params_pt, frozen_layers={0})
print("Freeze mask:")
for i, layer in enumerate(mask):
    print(f"  Layer {i}: w frozen={layer['w']}, b frozen={layer['b']}")

# Apply: keep frozen params unchanged, update the rest with the usual SGD step
def masked_update(param, grad, frozen):
    return jnp.where(frozen, param, param - lr * grad)

updated = jax.tree.map(masked_update, params_pt, grads, mask)
print(f"\nLayer 0 w changed? {not jnp.allclose(params_pt[0]['w'], updated[0]['w'])}")
print(f"Layer 1 w changed? {not jnp.allclose(params_pt[1]['w'], updated[1]['w'])}")
```

---

# 4. Summary

## Key Takeaways

- **Pytrees are JAX's universal data structure**: any nested combination of dicts, lists, tuples, and arrays. Machine learning model parameters are just large pytrees.

- **`jax.tree.map` replaces loops**: apply a function to every leaf array in a pytree simultaneously (e.g., applying a gradient update to every weight matrix in one line).

- **Transforms propagate through pytrees**: `jax.grad` of a function over a pytree returns a gradient pytree with the same structure. No manual destructuring.

- **Frameworks rely on pytrees**: Flax, Optax, and other JAX libraries all use pytrees to pass complex state through pure functions without losing the structure.

## What's Next

In **Notebook 06: Control Flow**, we'll write loops and conditionals that survive `jit` compilation: the structured primitives that replace Python `for`/`if` once you're inside a compiled function.

+++

---

# 5. Exercises

1. **Pytree inspection**: Create a simple dictionary containing two arrays, e.g., `{'a': jnp.array([1, 2]), 'b': jnp.array([3, 4, 5])}`. Use `jax.tree.leaves()` to see the flat list of arrays.

2. **Transforming a pytree**: Use `jax.tree.map` to multiply every array in your dictionary by 2.

3. **Two-tree map**: Build two pytrees with the same structure (e.g., two dicts with the same keys, each holding arrays). Use `jax.tree.map(lambda a, b: a + b, tree1, tree2)` to add them leaf-by-leaf. Verify the result matches adding each leaf manually.
