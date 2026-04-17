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

# Part 3: Autodiff

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

## A Note on the Example Model

Several sections below differentiate through a small **neural network** as a running example. You don't need to understand its internals yet (Notebook 07 builds one from scratch and explains every piece), but here's the one-paragraph version so the code isn't a black box:

- A **neural network** is just a function that turns an input vector into an output by passing it through a sequence of **layers**.

- One layer computes `x @ w + b`: a matrix multiply (`@` is Python's matrix-multiply operator) of the input `x` with a **weight** matrix `w`, plus a **bias** vector `b`. That's usually followed by a simple non-linear function such as **ReLU** (`jnp.maximum(x, 0)`, which replaces negatives with zero).

- An **MLP** (multi-layer perceptron) is the simplest kind of neural network: a stack of these layers. Our example MLP has layer sizes `8 → 64 → 32 → 1`, meaning an 8-number input is transformed step by step into a single output.

- The numbers inside every `w` and `b` are the model's **parameters** (also called **weights**). "Training" means adjusting them to make the output better, and that is exactly what gradients are for.

```{code-cell} ipython3
# @title MLP Utilities (used in Sections 3-4) { display-mode: "form" }

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

# Same architecture and seed used across notebooks
key = jax.random.key(42)
layer_sizes = [8, 64, 32, 1]
params = init_mlp_params(key, layer_sizes)

print("MLP reconstructed: ", " → ".join(str(s) for s in layer_sizes))
```

---

# 1. grad in 60 Seconds

[`jax.grad`](https://jax.readthedocs.io/en/latest/_autosummary/jax.grad.html) takes a function and returns a **new function** that computes the gradient.

```python
f  = lambda x: x ** 2    # A function
df = jax.grad(f)          # Another function - the derivative
```

Function in, function out. `grad` doesn't compute a gradient; it returns a new function that computes one when you call it.

> **What kind of differentiation?** This is **automatic differentiation** (autodiff), not symbolic differentiation (which manipulates math expressions like Wolfram Alpha) and not finite differences (which approximate `(f(x+ε) - f(x)) / ε`). Autodiff traces the actual computation your function performs and computes *exact* derivatives (to machine precision) in a single backward pass.

```{code-cell} ipython3
# A simple scalar function
def f(x):
    return x ** 3 - 2 * x ** 2 + x

# grad returns a NEW function - the derivative of f
df = jax.grad(f)

# Call it like any other function
x = 3.0
print(f"f({x})  = {f(x)}")       # 3^3 - 2*3^2 + 3 = 27 - 18 + 3 = 12
print(f"f'({x}) = {df(x)}")      # if you remember calculus: f'(x) = 3x^2 - 4x + 1

# We can compute the derivative by hand using the power rule
# (d/dx of x^n is n * x^(n-1)) and compare:
analytical = 3 * x**2 - 4 * x + 1
print(f"\nBy-hand derivative at x={x}: {analytical}")
print(f"jax.grad agrees? {jnp.isclose(df(x), analytical)}")
```

```{code-cell} ipython3
def finite_difference(f, x, eps=1e-5):
    """Approximate derivative using central differences."""
    return (f(x + eps) - f(x - eps)) / (2 * eps)

# Compare across several points
print(f"{'x':>6} | {'grad(f)(x)':>12} | {'finite diff':>12} | {'error':>12}")
print("-" * 52)
for x_val in [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]:
    grad_val = df(x_val).item()
    fd_val = float(finite_difference(f, x_val))
    error = abs(grad_val - fd_val)
    print(f"{x_val:>6.1f} | {grad_val:>12.6f} | {fd_val:>12.6f} | {error:>12.2e}")

print("\ngrad is exact (to machine precision), not an approximation.")
```

---

# 2. What grad Actually Does

When you call [`grad(f)`](https://jax.readthedocs.io/en/latest/_autosummary/jax.grad.html), JAX doesn't symbolically differentiate your code (no algebra is performed). It doesn't approximate with finite differences either. It performs **reverse-mode automatic differentiation**: it traces your function (the same tracing used by `jit`, see Notebook 01) to record every operation, then walks the recorded operations *backward*, applying the chain rule at each step to accumulate the derivative.

```text
   Forward pass: compute values left → right
   ──────────────────────────────────────────────►

      x ───► [ × w ] ───► a ───► [ sin ] ───► y ───► L

   ◄──────────────────────────────────────────────
   Backward pass: propagate gradients right → left (chain rule)

      ∂L/∂x  ◄──  ∂L/∂a · ∂a/∂x  ◄──  ∂L/∂y · ∂y/∂a  ◄──  1
```

The forward pass runs once to compute the value. The backward pass runs once to compute *every* input gradient at the same time, in a single traversal, regardless of how many inputs you have. That's why reverse-mode is cheap for loss functions: one scalar output, many parameter gradients, all computed in a single backward pass. (For our small MLP that's ~2,700 gradients; for a real transformer it's billions.)

Because [`grad`](https://jax.readthedocs.io/en/latest/_autosummary/jax.grad.html) returns a function, you can apply [`grad`](https://jax.readthedocs.io/en/latest/_autosummary/jax.grad.html) to *that* function to get higher-order derivatives.

```{code-cell} ipython3
# f(x) = sin(x)
# f'(x) = cos(x)
# f''(x) = -sin(x)
# f'''(x) = -cos(x)
# f''''(x) = sin(x)   ← back to the start

f = jnp.sin
df = jax.grad(f)
ddf = jax.grad(df)
dddf = jax.grad(ddf)
ddddf = jax.grad(dddf)

x = 1.0
print("Successive derivatives of sin(x) at x = 1.0:")
print(f"  f(x)      = {f(x):.6f}     (sin)")
print(f"  f'(x)     = {df(x):.6f}     (cos)")
print(f"  f''(x)    = {ddf(x):.6f}    (-sin)")
print(f"  f'''(x)   = {dddf(x):.6f}   (-cos)")
print(f"  f''''(x)  = {ddddf(x):.6f}    (sin again)")
print(f"\n  Full circle? {jnp.isclose(f(x), ddddf(x))}")
```

```{code-cell} ipython3
# @title Visualize: Function and Its Derivatives { display-mode: "form" }

x = jnp.linspace(-2 * jnp.pi, 2 * jnp.pi, 300)

# Compute derivatives at each point (grad works on scalars, so we vmap)
f = jnp.sin
df = jax.grad(f)
ddf = jax.grad(df)

# vmap lets us apply a scalar function across an array (preview of Notebook 04)
y0 = jax.vmap(f)(x)
y1 = jax.vmap(df)(x)
y2 = jax.vmap(ddf)(x)

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(x, y0, label="sin(x)", linewidth=2.5, color='#1f77b4')
ax.plot(x, y1, label="grad(sin)(x) = cos(x)", linewidth=2, color='#2ca02c', linestyle='--')
ax.plot(x, y2, label="grad(grad(sin))(x) = -sin(x)", linewidth=2, color='#d62728', linestyle=':')
ax.axhline(y=0, color='gray', linewidth=0.5)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Composing grad: Higher-Order Derivatives', fontweight='bold', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("grad(grad(sin)) gives -sin - no symbolic algebra needed, just function composition.")
```

## 2.1 Differentiating with Respect to Multiple Arguments

By default, [`grad`](https://jax.readthedocs.io/en/latest/_autosummary/jax.grad.html) differentiates with respect to the **first** argument. Use `argnums` to change this.

```{code-cell} ipython3
def weighted_sum(w, x, b):
    return jnp.sum(w * x) + b

w = jnp.array([1.0, 2.0, 3.0])
x = jnp.array([4.0, 5.0, 6.0])
b = 10.0

# Gradient w.r.t. first arg (w) - default
dw = jax.grad(weighted_sum, argnums=0)(w, x, b)
print(f"d/dw = {dw}  (should be x = [4, 5, 6])")

# Gradient w.r.t. second arg (x)
dx = jax.grad(weighted_sum, argnums=1)(w, x, b)
print(f"d/dx = {dx}  (should be w = [1, 2, 3])")

# Gradient w.r.t. third arg (b)
db = jax.grad(weighted_sum, argnums=2)(w, x, b)
print(f"d/db = {db}  (should be 1.0)")

# Multiple at once
dw, dx = jax.grad(weighted_sum, argnums=(0, 1))(w, x, b)
print(f"\nBoth: dw={dw}, dx={dx}")
```

---

# 3. value_and_grad

## 3.1 What is a Loss Function?

A **loss function** measures how wrong your model's predictions are: it takes the model's output and the correct answer, and returns a single number: the **loss**. Smaller loss = better predictions.

Training a neural network is the process of finding weights that make the loss as small as possible. The recipe is short:

1. Pick weights at random.

2. Compute the loss.

3. Compute the **gradient** of the loss with respect to each weight (the slope: which direction makes the loss bigger).

4. Nudge each weight a small amount in the *opposite* direction.

5. Repeat thousands of times.

Step 3 is what `jax.grad` does. That's why it's central to using JAX for ML.

## 3.2 value_and_grad

In practice, you almost always want both the loss value *and* its gradients. [`jax.value_and_grad`](https://jax.readthedocs.io/en/latest/_autosummary/jax.value_and_grad.html) computes both in a single forward+backward pass, more efficient than calling the function and its gradient separately.

> **Why not two separate calls?** `f(x)` followed by `grad(f)(x)` runs the forward pass twice. `value_and_grad` runs it once and piggybacks the value on the backward pass: half the work for the same result.

```{code-cell} ipython3
def mse_loss(params, x, y):
    """Mean squared error loss."""
    pred = mlp_forward(params, x)
    return jnp.mean((pred - y) ** 2)

# Generate some synthetic data
key = jax.random.key(0)
key, x_key, noise_key = jax.random.split(key, 3)
x_sample = jax.random.normal(x_key, (8,))
y_sample = jnp.array(1.5)  # Target

# Compute loss AND gradients in one call
loss_val, grads = jax.value_and_grad(mse_loss)(params, x_sample, y_sample)

print(f"Loss: {loss_val:.6f}")
print(f"\nGradient structure (mirrors params):")
for i, (dw, db) in enumerate(grads):
    print(f"  Layer {i}: dw shape {dw.shape}, db shape {db.shape}")

print("\nThe gradient has the SAME structure as the params.")
print("This is a preview of pytrees - covered properly in Notebook 05.")
```

## 3.3 has_aux: Returning Extra Information

Sometimes you want the loss function to return additional outputs (predictions, intermediate values). Use `has_aux=True` to tell JAX that the function returns `(loss, auxiliary_data)` and only differentiate the loss.

> **Mental model**: `has_aux=True` tells `grad` that your function returns `(loss, extras)`. `grad` differentiates only through `loss` and passes `extras` back to you unchanged; they don't affect the gradient computation.

```{code-cell} ipython3
def loss_with_predictions(params, x, y):
    """Returns (loss, predictions) - grad only applies to loss."""
    pred = mlp_forward(params, x)
    loss = jnp.mean((pred - y) ** 2)
    return loss, pred  # pred is auxiliary data

# has_aux=True tells JAX the second return value isn't part of the loss
(loss_val, pred), grads = jax.value_and_grad(loss_with_predictions, has_aux=True)(
    params, x_sample, y_sample
)

print(f"Loss:       {loss_val:.6f}")
print(f"Prediction: {pred.item():.6f}")
print(f"Target:     {y_sample.item():.6f}")
print(f"\nGradients computed w.r.t. loss only, pred passed through unchanged.")
```

---

# 4. Summary

## Key Takeaways

- **`jax.grad` differentiates Python code**: it returns a new function that computes the exact gradient (not a numerical approximation) of a scalar-valued function.

- **`jax.value_and_grad` is highly efficient**: when training, you almost always need both the loss and the gradients. This computes both in a single forward pass.

- **Differentiate w.r.t. specific arguments**: use `argnums` to specify which arguments to take gradients with respect to (usually the parameters, not the input data).

- **Secondary outputs**: use `has_aux=True` if your loss function needs to return auxiliary data (like metrics or state) without differentiating it.

## What's Next

In **Notebook 04: JIT and Vmap**, we'll add the other two core transformations, `jit` (compile your function for speed) and `vmap` (auto-vectorize over a batch axis), and see how they compose with `grad` to give you the `jit(vmap(grad(f)))` pipeline.

+++

---

# 5. Exercises

1. **Basic grad**: Define `f(x) = x**4 - 3*x**2 + x`. Use `jax.grad` to compute the derivative. Verify analytically: `f'(x) = 4x**3 - 6x + 1`. Check at `x = 2.0`.

2. **argnums**: Write a function `f(a, b) = jnp.sum(a * b)`. Use `argnums` to compute the gradient with respect to `b` only. What is it analytically?

3. **Higher-order**: Compute the second derivative of `jnp.cos(x)` using `jax.grad(jax.grad(...))`. Evaluate at `x = 0.5` and verify it equals `-jnp.cos(0.5)`.
