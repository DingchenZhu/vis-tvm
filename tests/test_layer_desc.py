import tvm
from tvm import relay

from vis_compiler.layer_desc import extract_layer_descs


def _infer(mod):
    return relay.transform.InferType()(mod)


def test_extract_conv_chain():
    x = relay.var("x", shape=(1, 1, 32, 32), dtype="float32")
    w1 = relay.var("w1", shape=(4, 1, 3, 3), dtype="float32")
    y = relay.nn.conv2d(x, w1, padding=(1, 1), channels=4, kernel_size=(3, 3))
    w2 = relay.var("w2", shape=(4, 4, 3, 3), dtype="float32")
    z = relay.nn.conv2d(y, w2, padding=(1, 1), channels=4, kernel_size=(3, 3))
    fn = relay.Function([x, w1, w2], z)
    mod = tvm.IRModule.from_expr(fn)
    mod = _infer(mod)
    layers = extract_layer_descs(mod)
    assert len(layers) == 2
    assert layers[0].op == "conv2d" and layers[0].cin == 1 and layers[0].cout == 4
    assert layers[1].cin == 4 and layers[1].cout == 4


def test_extract_deformable():
    x = relay.var("x", shape=(1, 8, 16, 16), dtype="float32")
    off = relay.var("off", shape=(1, 18, 16, 16), dtype="float32")
    w = relay.var("w", shape=(8, 8, 3, 3), dtype="float32")
    y = relay.nn.deformable_conv2d(
        x,
        off,
        w,
        strides=(1, 1),
        padding=(1, 1),
        dilation=(1, 1),
        deformable_groups=1,
        groups=1,
        channels=8,
        kernel_size=(3, 3),
    )
    fn = relay.Function([x, off, w], y)
    mod = tvm.IRModule.from_expr(fn)
    mod = _infer(mod)
    layers = extract_layer_descs(mod)
    assert len(layers) == 1
    assert layers[0].op == "deformable_conv2d"
    assert layers[0].deformable is True
