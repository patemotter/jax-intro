# JAX made ~~easy~~ less hard.

Welcome! If you're coming from PyTorch, NumPy, or TensorFlow, you are probably used to mutation and state. JAX throws the bulk of that out the window.

This repository is a hands-on guide to building the **mental models** that make JAX unique. We aren't just learning an API; we're trying to think in a new way about array programming.

---

### Who is this for?
**If you know basic Python (variables, functions, loops), you're ready.** 

You don't need a lot of experience to get started. We'll explain the concepts as we go. If you've used NumPy before, you'll have a head start, but if not, don't worry.

---

### The Roadmap
Each notebook builds on the last. Here’s what we’re diving into:

| # | Notebook | Summary |
|---|---|---|
| 01 | **[Foundations](01_foundations.ipynb)** | Why JAX bans mutation, how to handle "randomness" without losing your mind, and building your first neural network forward pass from scratch. |
| 02 | **[Autodiff](02_autodiff.ipynb)** | The magic of `grad`. Learn how JAX differentiates through almost any Python code, handles Jacobians, and lets you write custom gradients. |
| 03 | **[JIT Compilation](03_jit.ipynb)** | Making code go fast. We'll look at "tracing," XLA, and why your code sometimes throws a `TracerError` (and how to fix it). |
| 04 | **[vmap — Vectorization](04_vmap.ipynb)** | Stop writing for-loops over batches. `vmap` lets you write code for a single example and automatically scales it to thousands. |
| 05 | **[Pytrees](05_pytrees.ipynb)** | JAX's secret sauce for handling complex data (like model weights). We'll learn how to map functions over nested lists, dicts, and custom objects. |
| 06 | **[Training Pipelines](06_training.ipynb)** | The "Real World" transition. We go from raw JAX to using **Flax** (for layers) and **Optax** (for optimizers) to train a real model. |
| 07 | **[Control Flow](07_control_flow.ipynb)** | How to do `if` and `while` in compiled code. We'll build an autoregressive decoder with a KV-cache to see it in action. |
| 08 | **[Sharding](08_sharding.ipynb)** | Scale up. Learn how to spread your data and models across multiple CPUs, GPUs, or TPUs using JAX's modern Sharding API. |
| 09 | **[Performance](09_performance.ipynb)** | Squeezing out every drop of speed. Debugging bottlenecks, profiling, and using advanced tricks like "buffer donation." |
| 10 | **[Pallas (Custom Kernels)](10_pallas.ipynb)** | The deep end. Peek inside TPU hardware and write your own high-performance hardware kernels (like a fused Softmax) from scratch. |

---

### Getting Started (Using `uv`)

We use `uv` for package management because it’s fast and keeps things relatively clean and separate.

1. **Install uv**: If you don't have it, [get it here](https://docs.astral.sh/uv/getting-started/installation/).
2. **Clone & Run**:
   ```bash
   git clone https://github.com/patemotter/jax-intro.git
   cd jax-intro
   
   # This will set up everything and open Jupyter automatically
   uv run jupyter notebook <notebook_name>.ipynb
   ```
