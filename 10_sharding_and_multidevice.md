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

# Part 10: Sharding and Multi-Device

```{code-cell} ipython3
# @title Setup { display-mode: "form" }

# Pretend our single CPU is actually 8 separate devices, so the sharding
# examples below have something to distribute across. On a real multi-GPU
# or multi-chip TPU host you'd remove this line - JAX would already see
# the real devices.
import os
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'
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
print(f"JAX version:     {jax.__version__}")
print(f"Default backend: {jax.default_backend()}")
print(f"Devices:         {n_devices} × {devices[0].platform.upper()}")
for i, d in enumerate(devices):
    print(f"  [{i}] {d}")
```

---

# 1. The Multi-Device Mental Model

In JAX, every array lives on one or more devices. There is no separate "distributed mode" you switch into: the same code runs on 1 device or 1,000 devices. The difference is how you **shard** (distribute) your arrays.

> **The idea**: you declare how data should be distributed, and JAX + XLA figure out the communication (all-reduce, all-gather, etc.) automatically.

The two distribution choices for any given array are:

| Choice | What it means | Use Case |
|---|---|---|
| **Replicated** | Every device holds a full copy of the array | Small params, broadcast data |
| **Sharded** | The array is split into pieces, one per device | Large weights, large batches |

(You can also mix: shard along one axis, replicate along another. That falls out of the sharding spec; it's not a separate mode.)

```{code-cell} ipython3
x = jnp.ones((8, 8))
print(f"Default array devices: {x.devices()}")
print(f"Number of devices: {len(x.devices())}")

# You can check if an array is on a single device or sharded
print(f"\nSharding info: {x.sharding}")
```

---

# 2. Meshes

A **mesh** is a logical grid imposed on your physical devices. You name the axes, and then refer to those names when describing how to shard data.

> **Mental model**: think of a mesh as a coordinate system for your devices. If your mesh has a `'data'` axis of size 4, each device gets a coordinate along that axis (0, 1, 2, 3). A `PartitionSpec` then says "split this array axis across the `'data'` mesh axis", meaning device 0 gets slice 0, device 1 gets slice 1, etc.

The mesh doesn't move data; it just assigns names to device arrangements.

```text
  Physical devices:  [Device0, Device1, Device2, Device3]

  1D Mesh - data parallelism (one named axis):
  ┌──────────┬──────────┬──────────┬──────────┐
  │ Device0  │ Device1  │ Device2  │ Device3  │
  └──────────┴──────────┴──────────┴──────────┘
  axis 'data' ──────────────────────────────►

  2D Mesh - data + model parallelism (two named axes):
  ┌──────────┬──────────┐
  │ Device0  │ Device1  │  ◄─── axis 'model'
  ├──────────┼──────────┤
  │ Device2  │ Device3  │  ◄─── axis 'model'
  └──────────┴──────────┘
  axis 'data' ──────────►

  A PartitionSpec maps array axes onto mesh axes:
  PartitionSpec('data', None)  →  shard axis 0 over 'data', replicate axis 1
```

```{code-cell} ipython3
# 1D mesh: all devices along one axis named 'data'
mesh_1d = Mesh(np.array(devices), axis_names=('data',))
print(f"1D Mesh: {mesh_1d.shape}")
print(f"  Axis 'data': {mesh_1d.shape['data']} devices")

# 2D mesh (if we have >= 4 devices, otherwise demonstrate the concept)
if n_devices >= 4:
    device_grid = np.array(devices[:4]).reshape(2, 2)
    mesh_2d = Mesh(device_grid, axis_names=('data', 'model'))
    print(f"\n2D Mesh: {mesh_2d.shape}")
    print(f"  Axis 'data':  {mesh_2d.shape['data']} devices")
    print(f"  Axis 'model': {mesh_2d.shape['model']} devices")
else:
    print(f"\n(Only {n_devices} device(s) - 2D mesh examples will use 1D mesh)")
    print(f"All sharding concepts apply; with 1 device, sharding is a no-op.")

# The mesh we'll use throughout this notebook
mesh = Mesh(np.array(devices), axis_names=('data',))
print(f"\nWorking mesh: {mesh.shape}")
```

---

# 3. NamedSharding and PartitionSpec

A `PartitionSpec` declares how each axis of an array maps to mesh axes:

```python
P('data', None)  # Shard axis 0 across 'data', replicate axis 1
P(None, 'model') # Replicate axis 0, shard axis 1 across 'model'
P()              # Fully replicated (no sharding)
P('data')        # 1D array sharded across 'data'
```

`NamedSharding` combines a mesh with a partition spec to create a concrete sharding.

> **`None` vs. a mesh axis name**: `None` in a `PartitionSpec` position means "replicate this array axis, so every device gets a full copy." A mesh axis name means "split this array axis across those devices, so each device gets a different slice." A fully replicated array (`P()`) is identical on every device; a fully sharded array (`P('data', 'model')`) has no device holding a complete copy.

```{code-cell} ipython3
# Create a sharding: split axis 0 across 'data' devices
sharding_split = NamedSharding(mesh, P('data'))

# Create a sharding: fully replicated
sharding_replicated = NamedSharding(mesh, P())

# Apply shardings with jax.device_put
x = jnp.arange(64).reshape(8, 8).astype(jnp.float32)

x_split = jax.device_put(x, sharding_split)
x_replicated = jax.device_put(x, sharding_replicated)

print("Original array:")
print(x)
print(f"\nSplit across 'data' axis (P('data')):")
print(f"  Devices: {x_split.devices()}")
print(f"  Sharding: {x_split.sharding}")

print(f"\nFully replicated (P()):")
print(f"  Devices: {x_replicated.devices()}")
print(f"  Sharding: {x_replicated.sharding}")
```

```{code-cell} ipython3
# @title Visualizing Array Sharding { display-mode: "form" }

def visualize_sharding_simple(arr, title=""):
    """Simple visualization of how an array is distributed."""
    sharding = arr.sharding
    print(f"\n{title}")
    print(f"  Array shape: {arr.shape}")
    print(f"  Partition spec: {sharding.spec}")

    if hasattr(sharding, 'mesh'):
        print(f"  Mesh: {sharding.mesh.shape}")

    # Show which devices hold which data
    try:
        shard_shape = arr.sharding.shard_shape(arr.shape)
        print(f"  Shard shape (per device): {shard_shape}")
    except:
        print(f"  (Shard shape unavailable)")

# Demonstrate different shardings on a matrix
x = jnp.ones((8, 8))

shardings = {
    "Replicated P()": P(),
    "Split rows P('data', None)": P('data', None),
    "Split cols P(None, 'data')": P(None, 'data'),
}

for name, spec in shardings.items():
    sharded_x = jax.device_put(x, NamedSharding(mesh, spec))
    visualize_sharding_simple(sharded_x, title=name)
```

```{code-cell} ipython3
# JAX provides a built-in visualizer
x = jnp.ones((8, 8))

print("=== Replicated (P()) ===")
replicated = jax.device_put(x, NamedSharding(mesh, P()))
jax.debug.visualize_array_sharding(replicated)

print("\n=== Sharded rows (P('data', None)) ===")
sharded = jax.device_put(x, NamedSharding(mesh, P('data', None)))
jax.debug.visualize_array_sharding(sharded)

print("\nThis visualization shows which device holds which part of the array.")
print("This is the main tool for debugging sharding issues - use it whenever something is slower than expected.")
```

---

# 4. jit with Sharding Constraints

You can tell [`jit`](https://jax.readthedocs.io/en/latest/_autosummary/jax.jit.html) how inputs should be sharded and how outputs should be arranged. This lets XLA plan the computation and communication together.

```{code-cell} ipython3
# A sharded matrix multiply
def matmul(x, w):
    return x @ w

# Declare shardings: shard x rows across data, replicate weights
x = jnp.ones((8, 8))
w = jnp.ones((8, 8))

x_sharding = NamedSharding(mesh, P('data', None))  # Split batch dim
w_sharding = NamedSharding(mesh, P())               # Replicate weights
out_sharding = NamedSharding(mesh, P('data', None)) # Split output batch dim

sharded_matmul = jax.jit(
    matmul,
    in_shardings=(x_sharding, w_sharding),
    out_shardings=out_sharding,
)

x_sharded = jax.device_put(x, x_sharding)
w_replicated = jax.device_put(w, w_sharding)

result = sharded_matmul(x_sharded, w_replicated)
print(f"Input x:  {x.shape}, sharded rows across {n_devices} devices")
print(f"Weight w: {w.shape}, replicated")
print(f"Output:   {result.shape}, sharded rows across {n_devices} devices")
print(f"\nEach device computed its chunk of rows independently - no communication needed!")
```

## 4.1 with_sharding_constraint

Sometimes you want to control sharding *inside* a function, for example ensuring that intermediate activations are sharded correctly.

```{code-cell} ipython3
from jax.lax import with_sharding_constraint

def constrained_fn(x, w):
    # Ensure intermediate result is sharded along data axis
    h = x @ w
    h = with_sharding_constraint(h, NamedSharding(mesh, P('data', None)))
    return jax.nn.relu(h)

# JIT infers the full sharding strategy including the constraint
sharded_fn = jax.jit(constrained_fn)
result = sharded_fn(x_sharded, w_replicated)
print(f"Output sharding: {result.sharding.spec}")
print("The constraint guided XLA's sharding decisions for the intermediate value.")
```

---

# 5. Data Parallelism

Data parallelism is the most common distribution strategy: **replicate the model on every device, shard the batch**.

Each device processes its slice of the batch and computes a local gradient. Before the parameter update, those local gradients are **all-reduced**: a collective operation where every device sums (or averages) its value with every other device's value, and ends up with the same combined result. In JAX, this happens automatically when you shard the batch and replicate the params; you don't write the all-reduce by hand.

```
                  ┌─── batch ───┐                       params (replicated)
                  │             │                       ┌──────────┐
  Device 0   ◄────┤   shard 0   ├──► forward + loss ──► │ identical│ ──► grad_0
  Device 1   ◄────┤   shard 1   ├──► forward + loss ──► │ on every │ ──► grad_1
  Device 2   ◄────┤   shard 2   ├──► forward + loss ──► │  device  │ ──► grad_2
  Device 3   ◄────┤   shard 3   ├──► forward + loss ──► └──────────┘ ──► grad_3
                  └─────────────┘                                          │
                                                                           ▼
                                                  XLA inserts an all-reduce
                                                  (sum / mean across devices)
                                                                           │
                                                                           ▼
                                                  same averaged grads on every device
                                                                           │
                                                                           ▼
                                                  identical SGD step → new params
                                                  (still replicated)
```

| Strategy | What's distributed | Best when |
|---|---|---|
| **Data parallelism** | Batch (each device gets different data) | Model fits on one device; want throughput |
| **Tensor parallelism** | Weight matrices (each device holds a shard) | Model is too large for one device |
| **Pipeline parallelism** | Layers (each device runs different layers) | Very deep models; latency less critical |

> **What `jax.device_put(array, sharding)` actually does**: it doesn't just annotate the array; it physically sends different slices to different devices right now. After `device_put`, each device holds `array.shape[batch_axis] // n_devices` rows. You can verify this with `jax.debug.visualize_array_sharding`.

```{code-cell} ipython3
def init_mlp_pytree(key, layer_sizes):
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
    for i, layer in enumerate(params):
        x = x @ layer['w'] + layer['b']
        if i < len(params) - 1:
            x = jnp.maximum(x, 0)
    return x

# Replicate params, shard batch
replicated = NamedSharding(mesh, P())
batch_sharded = NamedSharding(mesh, P('data'))
batch_sharded_2d = NamedSharding(mesh, P('data', None))

mlp_params = init_mlp_pytree(jax.random.key(0), [8, 64, 32, 1])

# Replicate params across all devices
mlp_params = jax.device_put(mlp_params, replicated)

# Create a sharded batch
x_batch = jnp.ones((n_devices * 32, 8))  # Batch divisible by n_devices
y_batch = jnp.zeros(n_devices * 32)
x_batch = jax.device_put(x_batch, batch_sharded_2d)
y_batch = jax.device_put(y_batch, batch_sharded)

def loss_fn(params, x, y):
    preds = jax.vmap(lambda xi: mlp_forward_pytree(params, xi))(x).squeeze()
    return jnp.mean((preds - y) ** 2)

@jax.jit
def dp_train_step(params, x, y, lr=0.01):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    # Gradients are automatically all-reduced because params are replicated
    params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
    return params, loss

# Run a training step
new_params, loss = dp_train_step(mlp_params, x_batch, y_batch)
print(f"Loss: {loss.item():.4f}")
print(f"Params still replicated: {new_params[0]['w'].sharding.spec}")
print(f"\nGradients were all-reduced automatically because params are replicated.")
print(f"No manual all-reduce needed - the sharding spec tells XLA what to do.")
```

---

# 6. Model (Tensor) Parallelism

When a model's weights are too large for a single device, you **shard the weights** across devices. This is tensor parallelism: each device holds a slice of the weight matrix and computes a partial result.

For a linear layer `y = x @ W`:
- Shard `W` along the output dimension: each device holds `W[:, chunk]` and computes `y[:, chunk]`

- Shard `W` along the input dimension: each device holds `W[chunk, :]` and computes a partial sum → needs an all-reduce

The first approach (column-parallel) is simpler because outputs can be concatenated without communication.

```{code-cell} ipython3
# Column-parallel: shard weight along output dimension
# Each device computes y_chunk = x @ W_chunk

def tensor_parallel_linear(x, w, b):
    """Linear layer where w is sharded along axis 1 (output dim)."""
    y = x @ w + b
    return y

# Create a weight matrix sharded along the output dimension
d_in, d_out = 64, 128
w = jax.random.normal(jax.random.key(0), (d_in, d_out))
b = jnp.zeros(d_out)
x = jax.random.normal(jax.random.key(1), (d_in,))

# Shard weight columns and bias across devices
# With 1 device: this is a no-op. With N devices: each gets d_out/N columns.
w_sharded = jax.device_put(w, NamedSharding(mesh, P(None, 'data')))
b_sharded = jax.device_put(b, NamedSharding(mesh, P('data')))
x_replicated = jax.device_put(x, NamedSharding(mesh, P()))

result = jax.jit(tensor_parallel_linear)(x_replicated, w_sharded, b_sharded)

print(f"Weight shape: {w.shape}, sharded P(None, 'data')")
print(f"  Per-device shard: ({d_in}, {d_out // n_devices})")
print(f"Input: {x.shape}, replicated")
print(f"Output: {result.shape}, sharding: {result.sharding.spec}")
print(f"\nEach device computed {d_out // n_devices} output features independently.")
```

---

# 7. Debugging Sharding

Sharding bugs are subtle: the code runs and gives correct results, but is much slower than expected because XLA had to insert extra communication. Two operations you'll see in the trace:

- **all-reduce**: every device contributes one value, all devices end up with the sum/mean. Used for gradient averaging in data parallelism.

- **all-gather**: every device has a slice, all devices end up with the concatenated whole. Inserted automatically when an op needs more of the array than the current shard provides.

Either is fine when needed, but neither is free. Here's how to spot the unintended ones.

```{code-cell} ipython3
def check_sharding(name, arr):
    """Print sharding info for an array."""
    print(f"  {name:.<40s} shape={str(arr.shape):<15s} spec={arr.sharding.spec}")

print("Sharding audit:")
check_sharding("params[0]['w']", mlp_params[0]['w'])
check_sharding("x_batch", x_batch)
check_sharding("updated params[0]['w']", new_params[0]['w'])

print("\n✓ Params replicated (P()) - correct for data parallelism")
print("✓ Batch sharded (P('data', None)) - correct for data parallelism")
print("✓ Updated params still replicated - all-reduce happened correctly")
```

## 7.1 Common Sharding Mistakes

| Symptom | Likely Cause | Fix |
|---|---|---|
| Unexpectedly slow | Unnecessary all-gather before matmul | Check intermediate sharding |
| OOM on one device | Unsharded large array | Apply appropriate PartitionSpec |
| Correct but slow | Batch not divisible by n_devices | Pad batch to multiple of n_devices |
| Wrong results | Sharding mismatch between function and data | Audit with [`jax.debug.visualize_array_sharding`](https://jax.readthedocs.io/en/latest/_autosummary/jax.debug.visualize_array_sharding.html) |

+++

---

# 8. Example: Sharded Matrix Multiplication

Instead of a complex neural network, let's distribute a large 2D matrix multiplication across our devices.

```{code-cell} ipython3
# Reuse the 'data' mesh defined earlier in this notebook.
sharding = NamedSharding(mesh, P('data', None))

# Create large matrices and shard them along axis 0 ('data').
A = jax.device_put(jnp.ones((8192, 8192)), sharding)
B = jax.device_put(jnp.ones((8192, 8192)), sharding)

C = jnp.dot(A, B)
print(f"A sharding: {A.sharding.spec}")
print(f"B sharding: {B.sharding.spec}")
print(f"C sharding: {C.sharding.spec}")
print("Computation successful across devices.")
```

> **This convenience hides a cost.** Both `A` and `B` are sharded the same way (rows over `'data'`), but a matmul needs each output row to see *all* of `B`. XLA therefore inserts communication (an all-gather of `B`) automatically — exactly the kind of hidden cost Section 7 warned about. That's the lesson worth keeping: the computation *works* across devices no matter how you shard, but making it *fast* means choosing shardings that minimize this communication.

---

# 9. Summary

## Key Takeaways

- **JAX arrays live on devices**: there's no separate distributed mode. You declare shardings, JAX handles communication.

- **Mesh + PartitionSpec** is the modern sharding API. A mesh names device axes; a partition spec maps array axes to mesh axes.

- **Data parallelism** (replicate params, shard batch) is the simplest and most common strategy. Gradients are all-reduced automatically.

- **Tensor parallelism** (shard weight matrices) lets you run models too large for one device. Column-parallel is simplest.

- **[`jax.debug.visualize_array_sharding`](https://jax.readthedocs.io/en/latest/_autosummary/jax.debug.visualize_array_sharding.html)** is the main tool for debugging. If something is slow, check for unexpected all-gathers.

- **If you see `pmap` in older code**, it's the previous-generation multi-device API. `jax.sharding` (covered here) is what JAX recommends for new work: the same `jit`-compiled function runs correctly on 1 device or 1000.

## What's Next

In **Notebook 11: Performance and Profiling**, we'll make that code fast and measurable: how to benchmark JAX correctly and avoid the recompilation traps that quietly slow training down.

+++

---

# 10. Exercises

1. **Shard a vector**: Take a 1D array of length 64. Create a 1D mesh of all available devices and shard the vector across it. Use `jax.debug.visualize_array_sharding` to confirm.

2. **Replicate vs shard**: Create a `(8, 8)` array. Place one copy with `P()` (replicated) and another with `P('data', None)` (sharded by rows). Verify with `.sharding.spec` that they differ.

3. **Data-parallel step**: Modify `dp_train_step` to also return per-batch accuracy. Run it once and confirm the loss matches the version without accuracy.
