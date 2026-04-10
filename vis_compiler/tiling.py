"""
Tiling decisions from docs/unet_fsrcnn_tiling_and_codegen_guide.md.

Applied as a separate stage after LayerDesc extraction and *before* ISA emission.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from vis_compiler.layer_desc import LayerDesc


def _pick_w_tile(w_in: int) -> int:
    """Micro-tile width along W inside a macro tile (32 / 64 / 128)."""
    if w_in >= 128:
        return 128
    if w_in >= 64:
        return 64
    return 32


def _macro_w_tiles(w_in: int) -> List[Tuple[int, int, int]]:
    """
    (w_start, w_size, bas_addr_hint) for each horizontal macro tile.
    Guide §5.1: W=256 → two 128-wide halves; bas_addr for right half uses +288 in sd_codegen.
    Here we expose widths only; address policy lives in the emitter.
    """
    if w_in <= 128:
        return [(0, w_in, 0)]
    # Split into 128-wide chunks (documented pattern for 256-wide feature maps).
    chunks = []
    start = 0
    hint = 0
    while start < w_in:
        sz = min(128, w_in - start)
        chunks.append((start, sz, hint))
        start += sz
        # Match sd_codegen right-half input addressing step (magic from their script).
        hint += 288 if sz == 128 else sz * 2
    return chunks


@dataclass
class TilingPlan:
    """Per-layer tiling + ISA template indices (guide-aligned)."""

    layer_idx: int
    h_out_per_step: int  # output rows advanced per outer step (e.g. 2 for template A)
    load_total_num: int  # number of DataLoader blocks along H for one macro W tile
    padding_num: int
    line_buffer_rows: int  # transnum / rows loaded into line buffer per block (4 or 6)
    line_buffer_reshape: int
    w_macro_tiles: List[Tuple[int, int, int]]
    w_micro_tile: int
    cin_group: int
    cout_group: int
    weight_parall_mode: int
    weight_transnum_base: int  # for 3x3 baseline; emitter scales by cin groups
    read_mode: int
    use_bilinear_weights: int  # WeightLoader.is_bilinear_bicubic (0/1/2)
    ky_outer: int  # deformable: ky loop count (sd_sr uses 3)
    ic_inner: int  # deformable: ic groups per ky (sd_sr uses 2)
    notes: str = ""


def choose_tiling(layer: LayerDesc) -> TilingPlan:
    """
    Map LayerDesc → TilingPlan using the documented UNet / FSRCNN templates.

    Non-conv layers get degenerate plans (emitter may ignore).
    """
    if layer.op in ("relu", "prelu"):
        return TilingPlan(
            layer_idx=layer.idx,
            h_out_per_step=1,
            load_total_num=1,
            padding_num=0,
            line_buffer_rows=1,
            line_buffer_reshape=0,
            w_macro_tiles=[(0, layer.w_in, 0)],
            w_micro_tile=32,
            cin_group=1,
            cout_group=1,
            weight_parall_mode=0,
            weight_transnum_base=1,
            read_mode=0,
            use_bilinear_weights=0,
            ky_outer=1,
            ic_inner=1,
            notes="activation — no conv tiling",
        )

    if layer.op == "pool2d":
        # Pooling templates vary; default to spatial blocks of 4 rows.
        h = layer.h_in
        rows = max(1, h // 4)
        return TilingPlan(
            layer_idx=layer.idx,
            h_out_per_step=4,
            load_total_num=rows,
            padding_num=1,
            line_buffer_rows=4,
            line_buffer_reshape=0,
            w_macro_tiles=_macro_w_tiles(layer.w_in),
            w_micro_tile=_pick_w_tile(layer.w_in),
            cin_group=1,
            cout_group=8,
            weight_parall_mode=0,
            weight_transnum_base=1,
            read_mode=0,
            use_bilinear_weights=0,
            ky_outer=1,
            ic_inner=1,
            notes="pool2d — coarse template",
        )

    h_in = layer.h_in
    w_in = layer.w_in
    cin, cout = layer.cin, layer.cout
    k = layer.k_h

    # §3.1: deformable / bilinear uses 6-row line-buffer fills; plain conv uses 4.
    if layer.deformable:
        line_rows = 6
        read_mode = 0  # sr script uses 0 with OffsetLoader + bilinear weights; some layers use 1
        bilinear = 1
        ky_outer = 3 if k == 3 else 1
        ic_inner = 2 if cin > 1 else 1
        h_out_per_step = 4  # sd_sr mid_part: cal_total_num = 32//4
        load_total_num = max(1, (h_in // h_out_per_step) * ky_outer * ic_inner)
        padding_num = 1
    else:
        line_rows = 4
        read_mode = 0
        bilinear = 0
        ky_outer = 1
        ic_inner = 1
        # Template A/B: two output rows per outer sweep → H/2 loader iterations
        h_out_per_step = 2
        load_total_num = max(1, h_in // h_out_per_step)
        padding_num = 1

    # Channel grouping (guide §3.3)
    if cin <= 1:
        cin_group = 1
    elif cin <= 4:
        cin_group = 4
    elif cin <= 8:
        cin_group = 4
    else:
        cin_group = 8

    if cout <= 4:
        cout_g = 4
    elif cout <= 8:
        cout_g = 8
    else:
        cout_g = min(32, max(8, (cout + 7) // 8 * 8))

    w_micro = _pick_w_tile(w_in)
    macros = _macro_w_tiles(w_in)

    # WeightLoader transnum for 3x3, mode 0: 9 for cin=1 (sd_codegen layer 0)
    if k == 3 and cin == 1:
        wt = 9
    elif k == 3:
        wt = 9 * ((cin + cin_group - 1) // cin_group)
    else:
        wt = k * k

    wpar = 0 if cout <= 8 else 1

    return TilingPlan(
        layer_idx=layer.idx,
        h_out_per_step=h_out_per_step,
        load_total_num=load_total_num,
        padding_num=padding_num,
        line_buffer_rows=line_rows,
        line_buffer_reshape=0,
        w_macro_tiles=macros,
        w_micro_tile=w_micro,
        cin_group=cin_group,
        cout_group=cout_g,
        weight_parall_mode=wpar,
        weight_transnum_base=wt,
        read_mode=read_mode,
        use_bilinear_weights=bilinear,
        ky_outer=ky_outer,
        ic_inner=ic_inner,
        notes="deformable" if layer.deformable else "standard conv",
    )


def plan_all(layers: List[LayerDesc]) -> List[TilingPlan]:
    return [choose_tiling(L) for L in layers]
