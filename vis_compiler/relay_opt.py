"""Relay passes to normalize the graph for codegen-friendly traversal."""
from typing import Dict, Tuple

import tvm
from tvm import relay


def optimize_for_codegen(
    mod: tvm.ir.IRModule,
    params: Dict[str, tvm.nd.NDArray],
    opt_level: int = 2,
) -> Tuple[tvm.ir.IRModule, Dict[str, tvm.nd.NDArray]]:
    """
    Run conservative TVM passes: fold constants, simplify, canonicalize.

    DeformableConv2d is left as `nn.deformable_conv2d` — hardware is targeted
    via LayerDesc + ISA templates (OffsetLoader + bilinear WeightLoader), not
    by lowering this op to generic conv in TE.
    """
    with tvm.transform.PassContext(opt_level=opt_level):
        seq = tvm.transform.Sequential(
            [
                relay.transform.InferType(),
                relay.transform.FoldConstant(),
                relay.transform.SimplifyInference(),
                relay.transform.CanonicalizeOps(),
                relay.transform.EliminateCommonSubexpr(),
                relay.transform.InferType(),
            ]
        )
        out = seq(mod)
    return out, params
