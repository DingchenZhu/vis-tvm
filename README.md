# tvm-tiling

A **stage-based compiler scaffold** that brings deep learning models into **Apache TVM Relay**, extracts **hardware-oriented layer descriptions**, applies **tiling plans** aligned with a UNet / FSRCNN accelerator design, and emits **micro-instruction dictionaries** (ISA-level records plus a post-pass for dependencies and virtual registers).

The goal is to replace monolithic reference scripts (`references/sd_codegen.py`, `references/sd_sr_codegen.py`) with a pipeline you can extend per model or backend without copying thousands of lines.

## What the pipeline does

1. **Import** — ONNX or PyTorch (traced) → Relay (`vis_compiler/frontend.py`).
2. **Optimize** — Relay passes for a stable, codegen-friendly graph (`vis_compiler/relay_opt.py`). `nn.deformable_conv2d` is kept as-is (not lowered to generic conv).
3. **Extract** — Ordered `LayerDesc` list: conv2d, deformable_conv2d, pooling, activations (`vis_compiler/layer_desc.py`).
4. **Tile** — Per-layer `TilingPlan` from the design guide (`vis_compiler/tiling.py`).
5. **Emit** — `DataLoader` / `WeightLoader` / `OffsetLoader` / … dicts (`vis_compiler/emit/`).
6. **Finalize** — Dependency edges and `dest` / `src*` registers (`vis_compiler/emit/post_pass.py`), matching the shape of golden pseudo-code dumps.

For hardware background, tiling rules, and how the legacy scripts map to this design, see:

- [`docs/unet_fsrcnn_tiling_and_codegen_guide.md`](docs/unet_fsrcnn_tiling_and_codegen_guide.md) (primary design doc)
- [`docs/unet_fsrcnn_codegen_architecture_notes.md`](docs/unet_fsrcnn_codegen_architecture_notes.md)
- [`docs/vis_compiler_guide.md`](docs/vis_compiler_guide.md) (compiler usage and extension points)

## Requirements

- **Python 3** with **Apache TVM** (your build or source tree on `PYTHONPATH`).
- **numpy** (TVM Python depends on it).
- Optional: **onnx**, **torch**, **torchvision** (for ONNX import and FSRCNN demo).

Constant folding in TVM’s `FoldConstant` pass uses LLVM internally. This repo’s `optimize_for_codegen` **skips `FoldConstant` when `tvm.runtime.enabled("llvm")` is false**, so a no-LLVM TVM build can still run the rest of the pipeline. To force behavior, set `PipelineConfig(fold_constant=True)` or `fold_constant=False`.

## Setup

Point `PYTHONPATH` at your TVM Python package and this repository (adjust paths to your machine):

```bash
export PYTHONPATH=/path/to/tvm/python:/path/to/tvm-tiling
```

## Basic use: Python API

```python
from vis_compiler.frontend import load_onnx, onnx_input_names
from vis_compiler.pipeline import CompilerPipeline, PipelineConfig

# Example: ONNX
model_path = "model.onnx"
names = onnx_input_names(model_path)
mod, params = load_onnx(model_path, shape_dict={names[0]: (1, 1, 144, 256)})

cfg = PipelineConfig(
    dump_relay_path="out/relay.txt",           # optional
    dump_layers_path="out/layers.json",        # optional
    dump_instructions_path="out/pseudo.txt", # optional: one str(dict) per line
)
result = CompilerPipeline(cfg).run(mod, params)

# In memory
for layer, plan in zip(result.layers, result.tilings):
    print(layer.op, layer.idx, plan.notes)
for inst in result.instructions[:5]:
    print(inst["op_code"], inst.get("layer_idx"))
```

PyTorch:

```python
import torch
from vis_compiler.frontend import load_pytorch
from vis_compiler.pipeline import CompilerPipeline, PipelineConfig

# model + example_inputs for torch.jit.trace
mod, params = load_pytorch(model, example_inputs)
result = CompilerPipeline(PipelineConfig()).run(mod, params)
```

## Basic use: demo script

From the **repository root**:

```bash
export PYTHONPATH=/path/to/tvm/python:/path/to/tvm-tiling

# ONNX (default path: sibling ../tvm-design/USR_Net.onnx) + optional instruction dump
python3 scripts/run_compiler_demo.py --out ./output/demo --pseudo-out ./output/demo/usr_net_pseudo.txt

# Custom ONNX path
python3 scripts/run_compiler_demo.py --onnx /path/to/model.onnx --out ./output/demo

# FSRCNN (references/models_new_930.py): Relay + layers JSON; add --fsrcnn-pseudo-out for instructions
python3 scripts/run_compiler_demo.py --out ./output/demo --fsrcnn-pseudo-out ./output/demo/fsrcnn_pseudo.txt
```

Typical outputs under `--out`:

| Artifact | When |
|----------|------|
| `usr_net_relay.txt` | ONNX model found |
| `usr_net_layers.json` | ONNX model found |
| `fsrcnn_relay.txt` | PyTorch / FSRCNN trace succeeds |
| `fsrcnn_layers.json` | FSRCNN pipeline completes through extraction |
| Paths passed to `--pseudo-out` / `--fsrcnn-pseudo-out` | When set |

## Other scripts

- **`scripts/verify_golden_post_pass.py`** — Check golden pseudo-code files against the post-pass round-trip (see script docstring).
- **`scripts/dump_sd_inst_pseudo_code.py`** — Emit from `references/sd_sr_codegen.py` and run the same `finalize_instructions` as `vis_compiler` (for diffing against `golden/`).

## Tests

From the repo root with the same `PYTHONPATH`:

```bash
pytest tests/
```

## Repository layout (short)

| Path | Role |
|------|------|
| `vis_compiler/` | Compiler package (`frontend`, `relay_opt`, `layer_desc`, `tiling`, `pipeline`, `emit/`) |
| `scripts/` | CLI demos and golden verification helpers |
| `tests/` | Pytest suite |
| `docs/` | Design and user guides |
| `references/` | Legacy ISA wrappers and monolithic codegen (comparison / porting source) |
| `golden/` | Reference pseudo-instruction text dumps |

## License / upstream

TVM is Apache-licensed; this repo’s layout follows your internal accelerator codegen effort. Point TVM at your own checkout or release as needed.
