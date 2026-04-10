"""Deformable path emits OffsetLoader + bilinear WeightLoader (not lowered to conv)."""
from vis_compiler.emit import isa
from vis_compiler.emit.emitter import InstructionEmitter
from vis_compiler.layer_desc import LayerDesc
from vis_compiler.tiling import choose_tiling


def test_deformable_emits_offset_and_bilinear():
    layer = LayerDesc(
        op="deformable_conv2d",
        idx=0,
        h_in=32,
        w_in=32,
        cin=8,
        cout=8,
        k_h=3,
        k_w=3,
        deformable=True,
    )
    plan = choose_tiling(layer)
    em = InstructionEmitter()
    em.reset()
    em.emit_layer(layer, plan)
    ops = [c["op_code"] for c in isa.Inst.code_list]
    assert "QuantLoader" in ops
    assert "OffsetLoader" in ops
    bilinear_wl = [
        c
        for c in isa.Inst.code_list
        if c.get("op_code") == "WeightLoader" and c.get("is_bilinear_bicubic") == 1
    ]
    assert len(bilinear_wl) >= 1
