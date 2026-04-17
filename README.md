# JAX made ~~easy~~ less hard.

An interactive notebook-based introduction to [JAX](https://jax.readthedocs.io/). The series covers JAX from first principles (arrays and immutability through automatic differentiation, JIT compilation, sharding, and custom hardware kernels) without assuming prior experience with NumPy or any deep-learning framework.

Each notebook is self-contained and can be read in isolation, but they're ordered so that concepts build on each other.

## Prerequisites

- Working knowledge of Python (variables, functions, loops, classes).

- Notebooks are sized to run in seconds on CPU; no GPU or TPU required, but performance and parallelism benefits will be realized on accelerator hardware.

Familiarity with NumPy or a deep-learning framework is helpful but not required. Every concept used in code is introduced before it appears.

## Contents

| # | Notebook | Topic |
|---|---|---|
| 01 | [Arrays and Immutability](01_jax_arrays_and_immutability.ipynb) | JAX arrays, the `.at[]` update syntax, and why JAX functions must be pure. |
| 02 | [Broadcasting and Randomness](02_randomness_and_broadcasting.ipynb) | Shape alignment rules, and JAX's explicit PRNG key model. |
| 03 | [Autodiff](03_autodiff.ipynb) | `jax.grad` and `jax.value_and_grad` over arbitrary Python. |
| 04 | [JIT and Vmap](04_jit_and_vmap.ipynb) | XLA compilation, tracing, and automatic vectorization. |
| 05 | [Pytrees](05_pytrees.ipynb) | JAX's nested-container abstraction for model parameters and gradients. |
| 06 | [Control Flow](06_control_flow.ipynb) | `lax.scan`, `lax.cond`, `lax.fori_loop`, and `lax.while_loop` under JIT. |
| 07 | [Training From Scratch](07_training_from_scratch.ipynb) | A complete training loop in raw JAX, with no frameworks. |
| 08 | [Flax Basics](08_flax_basics.ipynb) | `nn.Module`, `init`/`apply`, and `@nn.compact` for clean model definitions. |
| 09 | [Optax Optimization](09_optax_optimization.ipynb) | Functional optimizers: Adam, AdamW, momentum, schedules, gradient clipping. |
| 10 | [Sharding and Multi-Device](10_sharding_and_multidevice.ipynb) | Distributing arrays and computation with `Mesh` and `NamedSharding`. |
| 11 | [Performance and Profiling](11_performance_and_profiling.ipynb) | Correct benchmarking, recompilation, precision/quantization, and AOT compilation. |
| 12 | [Pallas Kernels](12_pallas_kernels.ipynb) | Writing custom hardware kernels when XLA's automatic optimization isn't enough. |
| 13 | [Capstone - LLM Inference](13_inference_patterns.ipynb) | Applying 01-12 to a modern workload: precision, bucketing, KV-cache, and AOT. |

## Getting Started

The project uses [uv](https://docs.astral.sh/uv/) for dependency management. Install uv first, then:

```bash
git clone https://github.com/patemotter/jax-intro.git
cd jax-intro
uv run jupyter lab
```

JupyterLab will open in your browser. Open any notebook to begin.

### Editing Notes

The MyST-Markdown files (`01_*.md`, `02_*.md`, …) are the **source of truth**. Each is paired with an `.ipynb` via [jupytext](https://jupytext.readthedocs.io/); the `.ipynb` is a generated artifact that carries the executed outputs GitHub renders. Edit the `.md` and treat the `.ipynb` as build output.

**Build the notebooks from the markdown** — regenerate every `.ipynb` from its `.md` and run all cells:

```bash
uv run jupytext --to ipynb --execute [0-9]*.md
```

This is the canonical step after editing any `.md`. It always goes markdown → notebook, so the direction is explicit (unlike `jupytext --sync`, which picks a direction by file timestamp and can overwrite your `.md` if an `.ipynb` happens to be newer). The `[0-9]*.md` glob targets the numbered notebooks only — don't use `*.md`, which would also sweep in this README (a plain doc with no kernel) and error out.

To regenerate the `.ipynb` from the `.md` **without** re-running cells (keeping whatever outputs already exist), use `--update` instead of `--execute`:

```bash
uv run jupytext --to ipynb --update [0-9]*.md
```

- Editing in JupyterLab works too: saving updates both files. However you edit, commit the `.md` — that's what review happens against.

- After changing a code cell, rebuild with `--execute` (above) so the rendered outputs match the new code.

- **For contributors:** Run `git config core.hooksPath .githooks` once to enable the repository's pre-commit hook, which keeps the `.ipynb`/`.md` pair consistent and verifies outputs are present before committing.

## How to Use the Series

- **Reading order.** Notebooks 01-06 cover JAX core APIs and assume no machine-learning background. Notebooks 07-11 build a small neural network end-to-end. Notebook 12 (Pallas) is a low-level hardware deep dive. Notebook 13 (Capstone) is the culmination of the series, combining concepts from 01-11.

- **Running the code.** Every notebook is runnable top-to-bottom. Each one re-defines the small utilities it needs (no hidden state between notebooks), so you can jump in anywhere.

- **Exercises.** Most notebooks end with three short exercises. Solutions are not included; they're there for you to try.
