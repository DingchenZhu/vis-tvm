import tvm
from tvm import relay

from vis_compiler.pipeline import CompilerPipeline, PipelineConfig
from vis_compiler.relay_opt import optimize_for_codegen


def test_optimize_does_not_remove_deformable():
    x = relay.var("x", shape=(1, 4, 8, 8), dtype="float32")
    off = relay.var("off", shape=(1, 18, 8, 8), dtype="float32")
    w = relay.var("w", shape=(4, 4, 3, 3), dtype="float32")
    y = relay.nn.deformable_conv2d(
        x,
        off,
        w,
        strides=(1, 1),
        padding=(1, 1),
        dilation=(1, 1),
        deformable_groups=1,
        groups=1,
        channels=4,
        kernel_size=(3, 3),
    )
    fn = relay.Function([x, off, w], y)
    mod = tvm.IRModule.from_expr(fn)
    mod2, _ = optimize_for_codegen(mod, {})
    text = mod2.astext()
    assert "deformable_conv2d" in text


def test_pipeline_json_dump(tmp_path):
    x = relay.var("x", shape=(1, 1, 4, 4), dtype="float32")
    w = relay.var("w", shape=(2, 1, 1, 1), dtype="float32")
    y = relay.nn.conv2d(x, w, padding=(0, 0), channels=2, kernel_size=(1, 1))
    fn = relay.Function([x, w], y)
    mod = tvm.IRModule.from_expr(fn)
    cfg = PipelineConfig(
        run_optimize=True,
        dump_relay_path=str(tmp_path / "r.txt"),
        dump_layers_path=str(tmp_path / "layers.json"),
    )
    p = CompilerPipeline(cfg)
    res = p.run(mod, {})
    assert res.layers
    assert (tmp_path / "r.txt").is_file()
    assert (tmp_path / "layers.json").is_file()
