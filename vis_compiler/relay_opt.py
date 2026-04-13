"""Relay passes to normalize the graph for codegen-friendly traversal."""
from typing import Dict, Optional, Tuple

import tvm
from tvm import relay


def optimize_for_codegen(
    mod: tvm.ir.IRModule,
    params: Dict[str, tvm.nd.NDArray],
    opt_level: int = 2,
    fold_constant: Optional[bool] = None,
) -> Tuple[tvm.ir.IRModule, Dict[str, tvm.nd.NDArray]]:
    """
    Run conservative TVM passes: optionally fold constants, simplify, canonicalize.

    **FoldConstant** evaluates constant subgraphs on CPU using ``Target("llvm")``
    inside TVM (see ``tvm/src/relay/transforms/fold_constant.cc``). If your TVM
    build has ``target.build.llvm`` disabled, constant folding fails at runtime.
    For ``fold_constant=None`` (default), folding runs only when
    ``tvm.runtime.enabled("llvm")`` is true; otherwise it is skipped.

    DeformableConv2d is left as `nn.deformable_conv2d` — hardware is targeted
    via LayerDesc + ISA templates (OffsetLoader + bilinear WeightLoader), not
    by lowering this op to generic conv in TE.
    """
    if fold_constant is None:
        fold_constant = bool(tvm.runtime.enabled("llvm"))

    passes = [
        relay.transform.InferType(),
    ]
    if fold_constant:
        passes.append(relay.transform.FoldConstant())
    passes.extend(
        [
            relay.transform.SimplifyInference(),
            relay.transform.CanonicalizeOps(),
            relay.transform.EliminateCommonSubexpr(),
            relay.transform.InferType(),
        ]
    )

    with tvm.transform.PassContext(opt_level=opt_level):
        seq = tvm.transform.Sequential(passes)
        out = seq(mod)
    return out, params
