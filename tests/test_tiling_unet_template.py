"""Tiling matches guide §5.1 template A (256×144, layer0-style conv)."""
from vis_compiler.layer_desc import LayerDesc
from vis_compiler.tiling import choose_tiling


def test_template_a_spatial():
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
    p = choose_tiling(layer)
    assert p.h_out_per_step == 2
    assert p.load_total_num == 72
    assert p.padding_num == 1
    assert p.line_buffer_rows == 4
    assert len(p.w_macro_tiles) == 2
    assert p.w_macro_tiles[0][1] == 128 and p.w_macro_tiles[1][1] == 128
    assert p.weight_transnum_base == 9
    assert p.use_bilinear_weights == 0


def test_deformable_mid_shape_from_sr_script():
    """Aligns with sd_sr cal_total_num=32//4, ky=3, ic=2."""
    layer = LayerDesc(
        op="deformable_conv2d",
        idx=3,
        h_in=32,
        w_in=32,
        cin=8,
        cout=8,
        k_h=3,
        k_w=3,
        deformable=True,
    )
    p = choose_tiling(layer)
    assert p.line_buffer_rows == 6
    assert p.ky_outer == 3
    assert p.ic_inner == 2
    assert p.load_total_num == (32 // 4) * 3 * 2
    assert p.use_bilinear_weights == 1
