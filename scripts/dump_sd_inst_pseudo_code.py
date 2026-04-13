#!/usr/bin/env python3
"""
Emit ``sd_inst`` from ``references/sd_sr_codegen.py`` and write finalized pseudo-code
(one ``str(dict)`` per line), using the same post-pass as ``vis_compiler``.

Use this to compare against ``golden/pseudo_code_*.txt`` (note: those files are often
**slices** with a non-zero starting ``code_num`` and will not match a full dump
byte-for-byte without the same slice window).

  conda activate hhb
  export PYTHONPATH=/home/hansz/scratch-data/design/tvm/python:/home/hansz/scratch-data/design/tvm-tiling:references

  python scripts/dump_sd_inst_pseudo_code.py -o /tmp/sd_inst_first.txt --is-first --load-next
  python scripts/dump_sd_inst_pseudo_code.py -o /tmp/sd_inst_cont.txt --no-is-first --load-next
"""
from __future__ import annotations

import argparse
import copy
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", required=True, help="Output .txt path")
    parser.add_argument(
        "--is-first",
        action="store_true",
        help="Pass is_first=True into sd_inst (UNet bootstrap Offchip sequence)",
    )
    parser.add_argument(
        "--load-next",
        action="store_true",
        help="Pass load_next=True into sd_inst",
    )
    parser.add_argument(
        "--slice-from",
        type=int,
        default=None,
        help="Drop instructions before this index after emission",
    )
    parser.add_argument(
        "--slice-to",
        type=int,
        default=None,
        help="Drop instructions at this index and after (exclusive)",
    )
    args = parser.parse_args()

    ref = os.path.join(_ROOT, "references")
    if ref not in sys.path:
        sys.path.insert(0, ref)

    import instruction as instr  # noqa: E402
    import sd_sr_codegen as m  # noqa: E402

    instr.Inst.current_code_num = 0
    instr.Inst.code_list = []
    m.sd_inst(is_first=args.is_first, load_next=args.load_next)
    raw = copy.deepcopy(instr.Inst.code_list)

    from vis_compiler.emit.post_pass import finalize_instructions  # noqa: E402

    finalize_instructions(raw)

    lo = args.slice_from or 0
    hi = args.slice_to if args.slice_to is not None else len(raw)
    chunk = raw[lo:hi]

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for row in chunk:
            f.write(str(row) + "\n")
    print("Wrote", len(chunk), "lines to", args.output)


if __name__ == "__main__":
    main()
