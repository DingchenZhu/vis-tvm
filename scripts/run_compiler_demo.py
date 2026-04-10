#!/usr/bin/env python3
"""
Demo: optional ONNX + optional FSRCNN PyTorch → Relay dump + layer JSON + opcode counts.

Environment (per user):
  conda activate hhb
  export PYTHONPATH=/home/hansz/scratch-data/design/tvm/python:/home/hansz/scratch-data/design/tvm-tiling
"""
from __future__ import annotations

import argparse
import os
import sys

# tvm-tiling/ → design/ parent for default model paths
_TILING_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DESIGN_ROOT = os.path.dirname(_TILING_ROOT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--onnx",
        default=os.path.join(_DESIGN_ROOT, "tvm-design", "USR_Net.onnx"),
        help="Path to USR_Net.onnx (default: <design>/tvm-design/USR_Net.onnx)",
    )
    parser.add_argument("--out", default=os.path.join(_TILING_ROOT, "output", "demo"))
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    from vis_compiler.frontend import dump_relay, load_onnx, onnx_input_names
    from vis_compiler.pipeline import CompilerPipeline, PipelineConfig

    if os.path.isfile(args.onnx):
        print("Loading ONNX:", args.onnx)
        names = onnx_input_names(args.onnx)
        if not names:
            print("No ONNX inputs found")
        else:
            shape_dict = {names[0]: (1, 1, 144, 256)}
            mod, params = load_onnx(args.onnx, shape_dict=shape_dict)
            cfg = PipelineConfig(
                dump_relay_path=os.path.join(args.out, "usr_net_relay.txt"),
                dump_layers_path=os.path.join(args.out, "usr_net_layers.json"),
            )
            pipe = CompilerPipeline(cfg)
            res = pipe.run(mod, params)
            from collections import Counter

            c = Counter(i["op_code"] for i in res.instructions)
            print("Instruction opcode counts:", dict(c))
    else:
        print("ONNX not found, skip USR_Net:", args.onnx)

    try:
        import torch

        sys.path.insert(0, os.path.join(_TILING_ROOT, "references"))
        from models_new_930 import FSRCNN

        model = FSRCNN(2, num_channels=1, d=32, s=8, m=4).eval()
        x = torch.randn(1, 1, 32, 32)
        from vis_compiler.frontend import load_pytorch

        mod2, p2 = load_pytorch(model, x)
        dump_relay(mod2, os.path.join(args.out, "fsrcnn_relay.txt"))
        print("Wrote FSRCNN relay to", os.path.join(args.out, "fsrcnn_relay.txt"))
    except Exception as e:
        print("FSRCNN import/trace skipped:", e)


if __name__ == "__main__":
    main()
