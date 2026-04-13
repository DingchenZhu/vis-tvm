"""Dependency + register post-pass (golden-compatible instruction dicts)."""
import copy

from vis_compiler.emit import isa
from vis_compiler.emit.emitter import InstructionEmitter
from vis_compiler.emit.post_pass import finalize_instructions, strip_post_pass_fields
from vis_compiler.layer_desc import LayerDesc
from vis_compiler.tiling import choose_tiling


def test_finalize_smoke_pseudo_op():
    isa.reset_instruction_stream()
    isa.Inst.code_list.append(
        {
            "code_num": [0],
            "op_code": "PseudoOp",
            "note": "test",
        }
    )
    raw = list(isa.Inst.code_list)
    stats = finalize_instructions(raw)
    assert stats["num_instructions"] == 1
    assert raw[0]["dependency"] == []
    assert raw[0]["dest"] in range(1, 16)


def test_finalize_layer0_conv_emits_dependency_and_regs():
    layer = LayerDesc(
        op="conv2d",
        idx=0,
        h_in=144,
        w_in=256,
        cin=1,
        cout=4,
        k_h=3,
        k_w=3,
        groups=1,
    )
    plan = choose_tiling(layer)
    em = InstructionEmitter()
    em.reset()
    em.emit_layer(layer, plan)
    raw = list(isa.Inst.code_list)
    finalize_instructions(raw)
    assert len(raw) > 10
    for inst in raw:
        assert "dependency" in inst
        assert "dest" in inst
        assert "src1" in inst
    wl = [x for x in raw if x["op_code"] == "WeightLoader"]
    assert wl and wl[0].get("is_skip") == 2
    assert wl[0].get("is_new") == 1


def test_verify_instruction_list_roundtrip_on_emit():
    from vis_compiler.emit.post_pass import verify_instruction_list_roundtrip

    layer = LayerDesc(
        op="conv2d",
        idx=0,
        h_in=144,
        w_in=256,
        cin=1,
        cout=4,
        k_h=3,
        k_w=3,
        groups=1,
    )
    plan = choose_tiling(layer)
    em = InstructionEmitter()
    em.reset()
    em.emit_layer(layer, plan)
    raw = list(isa.Inst.code_list)
    finalize_instructions(raw)
    assert verify_instruction_list_roundtrip(raw)


def test_strip_and_refinalize_is_idempotent_for_emitted():
    layer = LayerDesc(
        op="conv2d",
        idx=0,
        h_in=144,
        w_in=256,
        cin=1,
        cout=4,
        k_h=3,
        k_w=3,
        groups=1,
    )
    plan = choose_tiling(layer)
    em = InstructionEmitter()
    em.reset()
    em.emit_layer(layer, plan)
    once = copy.deepcopy(list(isa.Inst.code_list))
    finalize_instructions(once)
    twice = copy.deepcopy(once)
    strip_post_pass_fields(twice)
    finalize_instructions(twice)
    assert once == twice
