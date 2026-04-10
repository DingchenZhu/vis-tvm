# VIS Compiler (`vis_compiler`) — User Guide

This document describes the TVM-based compiler scaffold under `tvm-tiling/vis_compiler/`: what it does, how to run it, how to inspect Relay IR and outputs, how **DeformableConv2d** is handled, how the design stays extensible, and **where to optimize and extend** it (§7–§9).

---

## 1. What Has Been Implemented

A **stage-based pipeline** (not a single monolithic script) that:

1. **Imports models** into **Relay IR**:
   - **ONNX** (e.g. USR_Net / U-Net style): `onnx.load` → `relay.frontend.from_onnx`, with a workaround for ONNX protobuf dtypes so TVM receives string dtypes like `"float32"`.
   - **PyTorch** (e.g. FSRCNN in `references/models_new_930.py`): `torch.jit.trace` → `relay.frontend.from_pytorch`.

2. **Runs Relay optimization passes** aimed at a stable, codegen-friendly graph: `InferType`, `FoldConstant`, `SimplifyInference`, `CanonicalizeOps`, `EliminateCommonSubexpr`, then `InferType` again (`relay_opt.py`). These passes do **not** remove or rewrite `nn.deformable_conv2d` into something else.

3. **Extracts a neutral layer list** (`LayerDesc`): conv2d, **deformable_conv2d**, pool2d, relu/prelu — ordered in a dependency-respecting way over the main function’s `Let` spine (`layer_desc.py`).

4. **Applies tiling decisions** *before* instruction emission (`tiling.py`), following the ideas in `unet_fsrcnn_tiling_and_codegen_guide.md` (e.g. H blocks, W macro tiles of 128 when width is large, 4-row vs 6-row line-buffer behavior, deformable branches with ky/ic-style grouping).

5. **Emits hardware micro-instructions** via the same ISA wrappers as the reference (`emit/isa.py`: `DataLoader`, `WeightLoader`, `OffsetLoader`, `QuantLoader`, `DataStorer`, etc.) and a pluggable **`InstructionEmitter`** (`emit/emitter.py`).

6. **Tests** under `tests/` validate tiling numbers, layer extraction, that optimization keeps deformable ops, deformable emission includes `OffsetLoader` and bilinear `WeightLoader`, and pipeline JSON/Relay dumps.

7. **Demo script** `scripts/run_compiler_demo.py` runs ONNX (if present) through the full pipeline and traces FSRCNN to a Relay text dump.

---

## 2. How to Use the Compiler

### Environment

Use your conda env and point `PYTHONPATH` at this repo’s TVM Python tree **and** `tvm-tiling` (so `vis_compiler` is importable):

```bash
conda activate hhb
export PYTHONPATH=/home/hansz/scratch-data/design/tvm/python:/home/hansz/scratch-data/design/tvm-tiling
cd /home/hansz/scratch-data/design/tvm-tiling
```

Adjust paths if your checkout lives elsewhere.

### Programmatic API

```python
from vis_compiler.frontend import load_onnx, load_pytorch, onnx_input_names, dump_relay
from vis_compiler.pipeline import CompilerPipeline, PipelineConfig

# ONNX: discover input name(s), then pass static shapes
names = onnx_input_names("/path/to/USR_Net.onnx")
mod, params = load_onnx(
    "/path/to/USR_Net.onnx",
    shape_dict={names[0]: (1, 1, 144, 256)},
)

cfg = PipelineConfig(
    run_optimize=True,
    dump_relay_path="out/relay_optimized.txt",      # optional
    dump_layers_path="out/layers_and_tiling.json",  # optional
)
pipe = CompilerPipeline(cfg)
result = pipe.run(mod, params)

# result.mod          — optimized IRModule
# result.params       — parameter dict
# result.layers       — list[LayerDesc]
# result.tilings      — list[TilingPlan] (one per layer)
# result.instructions — list of dicts (op_code, fields, …)
```

**PyTorch (FSRCNN):**

```python
import torch
from vis_compiler.frontend import load_pytorch, dump_relay

model = ...  # e.g. FSRCNN(...).eval()
x = torch.randn(1, 1, 32, 32)
mod, params = load_pytorch(model, x)
dump_relay(mod, "out/fsrcnn_relay.txt")
```

### CLI Demo

```bash
python scripts/run_compiler_demo.py --onnx /path/to/USR_Net.onnx --out output/demo
```

If the default ONNX path does not exist, the script skips USR_Net but still tries FSRCNN and writes `fsrcnn_relay.txt` under `--out`.

### Pipeline Hooks (extension point)

```python
def after_tile(res):
    print(len(res.layers), "layers")

pipe = CompilerPipeline(cfg)
pipe.add_hook("after_tile", after_tile)
res = pipe.run(mod, params)
```

Supported hook names: `after_opt`, `after_tile`, `after_emit`.

---

## 3. Dumping IR and Checking Correctness

### Relay text dump

- **Any time:** `dump_relay(mod, "path/to/file.txt")` writes `mod.astext(show_meta_data=False)`.
- **Through the pipeline:** set `PipelineConfig.dump_relay_path` to dump **after** the optimization sequence.

**What to check:**

- Operator set matches expectations (e.g. `nn.conv2d`, `nn.max_pool2d`, `concatenate`, etc. for UNet; `nn.deformable_conv2d` for FSRCNN mid blocks).
- Shapes in the printed IR are static where you supplied `shape_dict` / trace shapes.
- No unexpected fusion/removal of ops you care about (compare before/after opt by dumping twice with `run_optimize=False` vs `True`).

### Layer + tiling JSON

With `dump_layers_path` set, the pipeline writes a JSON array of objects combining each **`LayerDesc`** (as a dict) and its **`TilingPlan`** (`tiling` sub-object).

**What to check:**

- Number and order of layers vs your mental model of the network.
- Per-layer `h_in`, `w_in`, `cin`, `cout`, `k_h`, `k_w`, `groups`, `deformable` flags.
- Tiling fields: e.g. `load_total_num`, `line_buffer_rows`, `w_macro_tiles`, `use_bilinear_weights`, `ky_outer`, `ic_inner` for deformable-like plans.

### Instruction list

`result.instructions` is a Python list of dicts, one per emitted micro-op (`op_code`, `code_num`, and ISA fields).

**What to check:**

- Opcode histogram (e.g. count `DataLoader` / `WeightLoader` / `OffsetLoader`).
- For a small single-layer test, compare structure qualitatively to the reference scripts (`references/sd_codegen.py`, `references/sd_sr_codegen.py`).

### Automated tests

From `tvm-tiling`:

```bash
export PYTHONPATH=/path/to/tvm/python:/path/to/tvm-tiling
conda activate hhb
python -m pytest tests/ -v
```

If `pytest` is not installed, run the test modules’ functions manually (see `tests/*.py`).

---

## 4. DeformableConv2d — Detailed Handling

### Problem

TVM’s frontend can represent **`nn.deformable_conv2d`** in Relay, but a generic **TVM → TE → your VPU** lowering path may not map that op to your accelerator’s micro-ops. The hardware, as described in the architecture notes, instead uses **offset loading** and **bilinear/bicubic modes** on the load/MAC path (`OffsetLoader`, `WeightLoader` fields such as `is_bilinear_bicubic`, `offset_reg_idx`, and appropriate `read_mode` / line-buffer schedules in the handwritten SR codegen).

### Approach in This Compiler

1. **Import:** PyTorch uses `torchvision::deform_conv2d`; TVM’s PyTorch frontend lowers it to **`relay.op.nn.deformable_conv2d`** (three inputs: data, offset, weight). Confirmed in dumped FSRCNN Relay.

2. **Optimization:** Passes are chosen so they **do not** replace deformable conv with a long expanse of primitive ops for VPU lowering. The op remains **`nn.deformable_conv2d`** in Relay for this stack’s purposes.

3. **Layer extraction:** `extract_layer_descs()` detects `nn.deformable_conv2d` and builds a **`LayerDesc`** with `op="deformable_conv2d"`, `deformable=True`, and shapes/strides/padding read from Relay types and attributes — same as conv2d but with the deformable flag and `deformable_groups`.

4. **Tiling:** `choose_tiling()` switches to the **deformable-oriented** branch: larger **line-buffer row count (6)**, **`ky_outer` / `ic_inner`** style factors aligned with the SR reference loops, and **`use_bilinear_weights=1`** so the emitter knows to use the bilinear MAC path.

5. **Instruction emission:** `InstructionEmitter._emit_deformable_conv()` does **not** call TVM lowering. It emits a **micro-instruction sequence structurally inspired by `sd_sr_codegen.py`**: e.g. **`OffsetLoader.dispatch`** per ky step, **`DataLoader`** + **`WeightLoader`** with **`is_bilinear_bicubic=1`** and **`offset_reg_idx`** set, then **`DataStorer`** with pooling-related fields where the reference does. Exact addresses and inner loop counts remain **parameters** driven by `LayerDesc` + `TilingPlan`; they should be refined per layer as you harden the backend.

**Takeaway:** Deformable is handled as a **first-class logical op** in `LayerDesc` and a **dedicated ISA template** in the emitter, bypassing the missing TVM→VPU lowering for that op.

---

## 5. Why It Is Flexible — Codebase Structure

### Compared to `sd_codegen.py` / `sd_sr_codegen.py`

Those files embed **model topology, resolution, tiling, and addresses** in one long script. Here, responsibilities are split:

| Concern | Location |
|--------|----------|
| ISA field names / op records | `vis_compiler/emit/isa.py` |
| Relay import | `vis_compiler/frontend.py` |
| Relay passes | `vis_compiler/relay_opt.py` |
| Model-agnostic op + shape list | `vis_compiler/layer_desc.py` |
| Tiling policy (guide-driven) | `vis_compiler/tiling.py` |
| Relay → instructions | `vis_compiler/emit/emitter.py` |
| Orchestration + dumps + hooks | `vis_compiler/pipeline.py` |

You can **swap or extend** any stage without copying the whole codegen: e.g. new **`choose_tiling`** rules, new **`emit_*`** templates, or an extra Relay pass, without touching unrelated files.

### Extension patterns

- **New op in LayerDesc:** extend `_collect_calls_exec_order` handling / `extract_layer_descs` and add a branch in `InstructionEmitter.emit_layer`.
- **New tiling template:** add cases in `choose_tiling` or introduce a small registry keyed by `(op, H, W, Cin, Cout, ...)`.
- **New backend:** implement another emitter class with the same `LayerDesc` / `TilingPlan` inputs, or replace `emit_program` in the pipeline.

### Directory tree (essentials)

```
tvm-tiling/
  vis_compiler/
    __init__.py
    frontend.py          # ONNX / PyTorch → Relay
    relay_opt.py         # optimization pass sequence
    layer_desc.py        # Relay → LayerDesc
    tiling.py            # LayerDesc → TilingPlan
    pipeline.py          # CompilerPipeline, StageResult, hooks
    emit/
      __init__.py
      isa.py             # Inst, *Loader.dispatch
      emitter.py         # InstructionEmitter, emit_program
  tests/                 # unit checks
  scripts/
    run_compiler_demo.py
  docs/
    vis_compiler_guide.md    # this file
    unet_fsrcnn_*.md           # architecture + tiling reference
  references/            # instruction.py, models, sd_*_codegen.py
```

---

## 6. Additional Notes

### ONNX input names and dtypes

- Real ONNX graphs use their own input tensor names — use **`onnx_input_names()`** or inspect the model so `shape_dict` keys match.
- If `from_onnx` fails with dtype errors involving `numpy.float32`, **`load_onnx()`** already supplies a **per-input string dtype map** (`"float32"`). Override with `dtype=` if you need other types.

### USR_Net vs FSRCNN

- **USR_Net:** ONNX → Relay; pipeline produces many layers; current **standard conv emitter** is a **parameterized template** derived from early UNet-style loops — **deeper layers with smaller spatial sizes** need **template selection** per `LayerDesc` (next refinement).
- **FSRCNN:** PyTorch trace → Relay with explicit **`nn.deformable_conv2d`** nodes; deformable path is the focus of the custom emitter branch.

### Pseudo-ops

Activations (`relu`, `prelu`) and unmapped pool emission currently append a **`PseudoOp`** placeholder in the instruction list so the pipeline stays runnable; fuse them in Relay or map them to real ISA in a later emitter pass.

### Reference documents

- `docs/unet_fsrcnn_codegen_architecture_notes.md` — ISA roles, what was hard-coded in legacy codegen, target architecture for a general front-end.
- `docs/unet_fsrcnn_tiling_and_codegen_guide.md` — tiling vocabulary, W/H/IC/OC grouping, template narrative.

### Sanity checklist before trusting codegen

1. Relay dump looks right (ops + shapes).  
2. `layers_and_tiling.json` matches layer count and rough geometry.  
3. For deformable layers: instruction stream contains **`OffsetLoader`** and **`WeightLoader` with `is_bilinear_bicubic=1`**.  
4. Compare a **small** subgraph against hand-derived or reference Python codegen counts.

---

## 7. Further Optimization Opportunities

These are improvements to **compiler quality, maintainability, and confidence** — not necessarily “make the network faster on chip” (that is mostly tiling + hardware). Order is roughly **high leverage first**.

### 7.1 Correctness and observability

- **Unsupported Relay nodes are skipped silently** in `extract_layer_descs()`. Add an optional **diagnostic mode** (e.g. `PipelineConfig.warn_unsupported_ops=True` or a separate `relay_walk` report) that lists every `Call` op name not mapped to `LayerDesc`, with source span if available. That prevents “empty or partial layer list” surprises when you swap models.
- **Golden / structural tests against `references/sd_*.py`**: For one fixed small layer (e.g. single `conv2d` with known H/W/Cin/Cout), compare opcode sequences and critical fields (transnum, line_buffer, read_mode) to the handwritten reference. This catches regressions when you refactor the emitter.
- **Explicit “supported subset” manifest**: A single table (in code or doc) listing supported Relay ops and known limitations (NCHW, static batch, etc.) that tests or CI can cross-check.

### 7.2 Tiling and templates

- **Centralize magic numbers** (`288` step for W macro tiles, `h_in * 4` for storer base, padding row codes). Move them into a small **target layout / buffer geometry** dataclass (bytes per pixel, macro-tile width policy, alignment) so the emitter and `tiling.py` share one source of truth. Today they encode implicit assumptions from legacy scripts.
- **Template selection for standard conv**: The guide already notes that **deep / small spatial** layers need a different loop nest than the early UNet layer. Implement a **registry**: `choose_tiling` (or a sibling) picks `(template_id, TilingPlan)` from `(h_in, w_in, k, cin, cout, stride, groups)` and dispatch `emit_standard_conv_variant` in the emitter. Start with 2 templates (e.g. “full H sweep” vs “small feature map”) before generalizing.
- **Deformable path**: `load_total_num` and inner address increments are simplified; profile against `sd_sr_codegen.py` per layer geometry and add **unit tests** that pin expected loop counts for FSRCNN-shaped layers.

### 7.3 Pipeline and IR stages

- **Configurable Relay pass lists**: Expose pass sequences in `PipelineConfig` (or a named preset: `preserve_deformable`, `aggressive_fold`, etc.) so frontends can opt in/out without editing `relay_opt.py`.
- **Early failure for unsupported graphs**: If diagnostics show critical ops (e.g. `nn.batch_matmul`) with no lowering plan, optionally **abort with a clear error** instead of emitting instructions for a truncated layer list.
- **Performance of the Python pipeline**: For very large Relay graphs, repeated walks are cheap compared to TVM; if needed, cache `InferType` results or walk once and reuse. Low priority unless profiling shows a bottleneck.

### 7.4 Code structure

- **Emitter decomposition**: Split `InstructionEmitter` into mixins or per-template classes (`TemplateAEmitter`, `DeformableEmitter`) behind a thin facade to keep files small and tests targeted.
- **Instruction serialization**: If downstream tools consume JSON, add optional **schema versioning** and stable field ordering for diff-friendly dumps.

---

## 8. Development Advice — Making Features More Complete

Use these principles when growing from a **scaffold** toward a **trustworthy** compiler for your accelerator.

### 8.1 Represent the graph, not only a linear layer list

The current `LayerDesc` list is a **flattened spine** of supported conv/pool/activations. Real models need **residuals, concatenation, multi-output branches, and epilogue fusions**. Longer term:

- Either **lift** `LayerDesc` into a light **DAG** (nodes = compute, edges = tensor consumers) still derived from Relay, **or**
- Keep a linear schedule but add **explicit buffer read/write metadata** (which SRAM / DRAM base, lifetime) so later passes can schedule memory.

Without some graph or schedule structure, skip connections and in-place semantics stay invisible to codegen.

### 8.2 Separate “what” from “how”

- **What**: Relay types, shapes, attrs, and maybe TIR hints — the **semantic** view.
- **How**: `TilingPlan` + target descriptor — the **mapping** to line buffers, DMA strides, and ISA fields.

Avoid encoding buffer addresses only inside the emitter; thread them from a **memory planner** once you have multiple tensors live.

### 8.3 Grow the operator set deliberately

For each new Relay op (e.g. `add`, `concatenate`, `nn.batch_flatten`, depthwise-as-conv):

1. Define its **LayerDesc** (or subgraph pattern) and **tests** with a minimal Relay module.
2. Decide **tiling** (often “epilogue” or “pass-through” for elementwise).
3. Emit **real ISA** or a documented **PseudoOp** with enough fields for a later lowering pass.

Activations and pooling are currently **PseudoOp** placeholders; replacing them with real micro-ops is usually required before silicon-level confidence.

### 8.4 Frontends and dynamism

- **ONNX / PyTorch** remain the primary path; consider **TorchScript `export` / `dynamo`** when `trace` is too fragile.
- **Dynamic shapes**: Either reject explicitly in config or introduce a **symbolic** shape layer (TVM’s shape func) — a large project; document “static shapes only” until then.
- **Batch > 1**: Often a mechanical extension if the hardware uses batch-outer loops; thread `N` through `LayerDesc` and tiling once semantics are fixed.

### 8.5 Verification strategy

- **Level 1 — Structural**: instruction counts and opcode histograms vs reference scripts.
- **Level 2 — Functional**: if you have a **C/RTL simulator** or golden vector per layer, drive it from emitted binaries or JSON.
- **Level 3 — Numeric**: compare accelerator output to Relay/TVM reference on CPU for small tensors (may require a **bit-accurate** quant mode).

Start Level 1 before investing in full-network numeric tests.

### 8.6 Target description file

Introduce a **YAML or JSON “target”** (line buffer depth, max W tile, supported kernels, DMA alignment). `choose_tiling` and the emitter read it instead of hard-coded caps. That is how one codebase supports multiple chip stepping’s or FPGA vs ASIC.

---

## 9. Roadmap — Phased Plan

A practical sequence that balances **risk** and **usable milestones**. Adjust phases to your team size and tapeout pressure.

| Phase | Focus | Deliverables |
|-------|--------|----------------|
| **A — Diagnostics & contracts** | Know what the compiler ignores | Unsupported-op report; documented supported-op matrix; optional strict mode that errors on unmapped critical ops; expand pytest with “import real ONNX → Relay dump” smoke tests |
| **B — Multi-template standard conv** | USR_Net-style depth + resolution diversity | Template registry; at least two standard-conv emit paths tested against `sd_codegen` fragments; `layers_and_tiling.json` includes `template_id` |
| **C — Real epilogues** | Pool / activation / bias on hardware | Replace PseudoOp for pool and relu/prelu where ISA exists; or fuse into conv in Relay and document fusion rules |
| **D — Graph-aware scheduling** | Residuals and concat | Minimal DAG or explicit buffer binding between `LayerDesc` nodes; memory planner stub (allocate `a`/`b`/`offchip` regions) |
| **E — Deformable hardening** | FSRCNN-class SR models | Per-layer address/loop refinement vs `sd_sr_codegen.py`; golden instruction snippets per block type |
| **F — Quantization & modes** | Production weights | Wire `QuantLoader` / modes to exported calib tables; test int8 vs fp16 paths if hardware supports them |
| **G — End-to-end automation** | Developer UX | CLI: `vis-compile --target vis.yaml --onnx m.onnx --out bundle/`; CI: lint + tests + one reference model |

**Suggested order:** A → B → C runs in parallel with small E fixes if deformable is on the critical path; D before you rely on codegen for full UNet-like skip topology; F/G when approaching system integration.

---

*Last updated to match the `vis_compiler` package as implemented under `tvm-tiling/`, including optimization notes, development advice, and roadmap (§7–§9).*
