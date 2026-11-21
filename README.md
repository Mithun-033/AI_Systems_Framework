# AI Systems Framework

GPU-oriented training primitives implemented with PyTorch and Triton. The
repository currently contains matmult kernels, custom optimizers and loss funcs plus a
Triton implementation of the Muon optimizer's Newton-Schulz matrix
transformation with three kernel launch.

## What is implemented

- `TritonSGD`, `TritonAdam`, and `TritonAdamW`, exposed from
  [`src/optimizers.py`](src/optimizers.py).
- `TritonMSELoss`, `TritonCrossEntropyLoss`, and
  `TritonBCEWithLogitsLoss`, exposed from
  [`src/losses.py`](src/losses.py).
- `TritonMuon`, a Muon optimizer implementation in
  [`src/muon.py`](src/muon.py).
- Focused correctness tests in
  [`src/test_core_primitives.py`](src/test_core_primitives.py).

The optimizer and loss wrappers preserve PyTorch-style interfaces where
possible. CUDA tensors dispatch to Triton kernels; CPU tensors use explicit
PyTorch reference implementations for the supported operations. Optimizer
validation covers learning-rate and momentum constraints, invalid optimizer
names, missing gradients, and Adam state initialization.

## Muon kernel architecture

`TritonMuon.step()` maintains a momentum tensor per parameter, optionally
applies Nesterov momentum, casts the working matrix to `bfloat16`, normalizes
it, and performs Newton-Schulz iterations. A tall matrix is transposed into
the kernel-friendly orientation and transposed back before the parameter
update.

Each Newton-Schulz iteration is fused into three Triton launches:

1. **Gram product:** `A = X @ X.T`.
2. **Polynomial transform:** `out_2 = bA + c(A @ A.T)`.
3. **Rectangular reconstruction:** `out = aX + out_2 @ X`.

The first two kernels exploit symmetry: only lower-triangular tiles are
computed, and off-diagonal results are written to the reflected tile using a
transpose. The kernels use `64 x 64` output tiles, `32`-element reduction
tiles, grouped program IDs, masked loads/stores for ragged boundaries, and
FP32 accumulation. The third kernel uses `64 x 64` output tiles because its
result is rectangular and cannot use the symmetric-write optimization.

The default Newton-Schulz coefficients are `(3.445, -4.775, 2.0315)`,
with five iterations, `beta=0.95`, and `eps=1e-7`. Learning-rate scaling is
selectable through `adjust_lr_fn`: `"original"`,
`"match_rms_adamw"`, or `"spectral_unclamped"`.

## Installation

The project targets Python 3.13 or newer and uses `uv`:

```powershell
uv sync
```

The dependency set is defined in
[`pyproject.toml`](pyproject.toml).
It includes a CUDA 12.8 PyTorch index and `triton-windows`; a compatible
NVIDIA driver and CUDA-capable GPU are required for Triton tests and
benchmarks.

## Usage

```python
import torch

from src.losses import TritonCrossEntropyLoss
from src.optimizers import TritonAdamW

model = torch.nn.Linear(16, 10).cuda()
optimizer = TritonAdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
criterion = TritonCrossEntropyLoss()

logits = model(torch.randn(32, 16, device="cuda"))
target = torch.randint(0, 10, (32,), device="cuda")
loss = criterion(logits, target)
loss.backward()
optimizer.step()
optimizer.zero_grad()
```

Muon can be constructed directly:

```python
from src.muon import TritonMuon

optimizer = TritonMuon(
    model.parameters(),
    lr=1e-3,
    weight_decay=0.1,
    ns_steps=5,
    nesterov=True,
)
```

## Validation

Run the focused test module with:

```powershell
uv run python -m unittest src.test_core_primitives -v
```

CUDA-specific tests skip when CUDA is unavailable. The tests compare custom
losses and AdamW updates against PyTorch, and verify that all three
optimizers reduce a scalar quadratic objective. Numerical checks reported in
the matrix-multiplication development history used both absolute and relative
tolerances.

## Four-week engineering progress

The weeks below are a milestone-based reconstruction of the commit history,
not claims about daily activity. The repository history spans
2026-10-12 through 2026-11-21; commits are grouped into four engineering
phases.

### Week 1 — Kernel bring-up and tiling

- Bootstrapped the Python/CUDA environment in `8c1d923`.
- Implemented the standard tiled matrix-multiplication kernel in `5461da5`.
- Added the Gram/symmetric multiplication path in `dabce17`, including thread
  block swizzling and reflected dual stores.
- Added a split-K path for tall matrices and made launch-shape constants
  compile-time Triton parameters in `6e61415`.

The main design problem was not just arithmetic throughput: program ordering
had to improve L2 reuse without breaking boundary masking or the symmetric
output invariant. The Gram path therefore computes one triangular half and
materializes the transpose rather than launching redundant work.

### Week 2 — Correctness and occupancy debugging

- Added numerical coverage and fixed correctness issues across all three
  kernels in `1d5fc0d`.
- Benchmarked FP32 and BF16 behavior in `5624936`.
- Removed an unnecessary `c.zero_()` memset launch for tall matrices in
  `2b26745`.
- Fixed a fatal `group_size_k` configuration bug in `dcf5744`, changing the
  value from `64` to `512`.

The split-K design was not uniformly beneficial: the history records that it
underperformed substantially while the standard and Gram kernels were usually
faster than PyTorch matmul. The occupancy fix moved the affected kernel from
approximately `0.5x` Torch throughput to roughly `1.01–1.05x` in the recorded
benchmark cases. This was a launch-configuration failure, not an algorithmic
correctness failure.

### Week 3 — Differentiation and memory behavior

- Implemented the upstream-gradient contribution for `dA` in `29464de`.
- Added the corresponding `dB` gradient path in `ec642ea`.
- Completed the backward kernel for `A @ A.T` in `da723ea`.
- Reverted a lambda-based grid launch to the standard launch form in
  `ecd375b`.
- Improved the Gram transpose tile's memory access pattern in `fdbab6d`.
- Integrated the optimizer API surface in `f255174`, including `TritonSGD`,
  `TritonAdam`, `TritonAdamW`, parameter validation, momentum state, Adam
  bias correction, weight decay, and the optimizer factory.

This phase exposed two classes of bugs: gradient formulas had to preserve the
same tile and transpose semantics as the forward path, and launch indirection
made the kernel interface less predictable. The final implementation favors
explicit Triton grid launches and coalesced loads for the transposed operand.

### Week 4 — API integration and Muon delivery

- Stabilized and benchmarked the matmul API in `aad2cdb`.
- Completed the loss API in `f255174`: MSE, cross-entropy, and
  BCE-with-logits, with `none`, `mean`, and `sum` reductions where supported,
  shape validation, optional class/example weights, and CPU fallback
  behavior.
- Added the three-launch Muon implementation in `e8defa2`.
- The final Muon commit reports numerical agreement under both absolute and
  relative tolerance checks and a `1.2–1.4x` speedup over
  `torch.compile` for model sizes above approximately 10M parameters.
