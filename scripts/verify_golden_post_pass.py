#!/usr/bin/env python3
"""
Verify post-pass against golden/pseudo_code_*.txt (same env as compiler).

  conda activate hhb
  export PYTHONPATH=/home/hansz/scratch-data/design/tvm/python:/home/hansz/scratch-data/design/tvm-tiling

  python scripts/verify_golden_post_pass.py
  python scripts/verify_golden_post_pass.py golden/pseudo_code_load_next_mid.txt
"""
from __future__ import annotations

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_GOLDEN = [
    os.path.join(_ROOT, "golden", "pseudo_code_load_next_first.txt"),
    os.path.join(_ROOT, "golden", "pseudo_code_load_next_mid.txt"),
    os.path.join(_ROOT, "golden", "pseudo_code_load_next_last.txt"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        default=_DEFAULT_GOLDEN,
        help="Golden pseudo_code .txt files (default: all three under golden/)",
    )
    args = parser.parse_args()

    from vis_compiler.emit.post_pass import verify_golden_post_pass

    ok_all = True
    for p in args.paths:
        ok = verify_golden_post_pass(p)
        if ok is None:
            print("SKIP (starts at code_num > 0; sliced dump — use dump_sd_inst_pseudo_code.py):", p)
            continue
        ok_all = ok_all and ok
        print(("OK  " if ok else "FAIL"), p)
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
