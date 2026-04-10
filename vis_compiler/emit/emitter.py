"""
Relay / LayerDesc → hardware micro-instructions.

Standard conv uses the parameterized Template A pattern from sd_codegen.py
(layer 0, 256×144, left then right 128-wide macro tiles).

Deformable conv does **not** go through TVM lowering: we emit OffsetLoader +
WeightLoader(is_bilinear_bicubic=1) sequences matching sd_sr_codegen.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from vis_compiler.emit import isa
from vis_compiler.layer_desc import LayerDesc
from vis_compiler.tiling import TilingPlan


@dataclass
class EmitterState:
    """Mutable buffer / address state (extensible)."""

    dataloader_bas_addr: int = 0
    weight_bas_addr: List[int] = field(default_factory=lambda: [0, 0, 0])
    quant_bas_addr: int = 0
    storer_bas_addr: int = 0
    line_buffer_idx: int = 0
    acc_reg_idx: int = 0
    quant_config_idx: int = 0
    offset_reg_idx: int = 0


class InstructionEmitter:
    """
    Pluggable backend: subclass or replace methods for new templates.

    `emit_layer` dispatches on LayerDesc.op + tiling flags.
    """

    def __init__(self, state: Optional[EmitterState] = None):
        self.state = state or EmitterState()

    def reset(self) -> None:
        isa.reset_instruction_stream()
        self.state = EmitterState()

    def emit_layer(self, layer: LayerDesc, plan: TilingPlan) -> None:
        if layer.op == "deformable_conv2d":
            self._emit_deformable_conv(layer, plan)
        elif layer.op == "conv2d":
            self._emit_standard_conv(layer, plan)
        elif layer.op in ("relu", "prelu", "pool2d"):
            # Fused or no-op at ISA level in current scripts; record placeholder.
            isa.Inst.code_list.append(
                {
                    "code_num": [isa.Inst.current_code_num],
                    "op_code": "PseudoOp",
                    "note": f"skipped-{layer.op}",
                    "layer_idx": layer.idx,
                }
            )
            isa.Inst.current_code_num += 1

    def emit_quant_loader(self, layer_idx: int, transnum: int, quant_mode: int = 0) -> None:
        isa.QuantLoader.dispatch(
            quant_reg_load_idx=self.state.quant_config_idx,
            quant_mode=quant_mode,
            layer_idx=layer_idx,
            transnum=transnum,
            bas_addr=self.state.quant_bas_addr,
        )
        self.state.quant_bas_addr += transnum

    def _emit_standard_conv(self, layer: LayerDesc, plan: TilingPlan) -> None:
        """Template A-style loop when dimensions match UNet layer0 (guide §5.1)."""
        self.emit_quant_loader(layer.idx, transnum=4, quant_mode=0)
        for macro_idx, (w0, w_sz, bas_hint) in enumerate(plan.w_macro_tiles):
            self._emit_w_macro_tile(layer, plan, w0, w_sz, bas_hint, macro_idx)

    def _emit_w_macro_tile(
        self,
        layer: LayerDesc,
        plan: TilingPlan,
        w0: int,
        w_sz: int,
        bas_hint: int,
        macro_idx: int,
    ) -> None:
        st = self.state
        load_total = plan.load_total_num
        padding_num = plan.padding_num
        st.dataloader_bas_addr = bas_hint
        if macro_idx == 0:
            st.storer_bas_addr = 0
        else:
            # Right half output base from sd_codegen: 144*4 for second 128-wide tile output
            st.storer_bas_addr = layer.h_in * 4

        for load_idx in range(load_total):
            if load_idx < padding_num:
                is_padding_row = 1
            elif load_idx > load_total - 1 - padding_num:
                is_padding_row = 5
            else:
                is_padding_row = 0

            isa.DataLoader.dispatch(
                layer_idx=layer.idx,
                line_buffer_reshape=plan.line_buffer_reshape,
                is_padding_row=is_padding_row,
                read_mode=plan.read_mode,
                transnum=plan.line_buffer_rows,
                line_buffer_idx=st.line_buffer_idx,
                src_buffer_idx="offchip_input_buffer" if layer.idx == 0 else "a",
                bas_addr=st.dataloader_bas_addr,
            )
            st.line_buffer_idx = 1 - st.line_buffer_idx
            st.dataloader_bas_addr += 2 if load_idx < padding_num else 4

            isa.WeightLoader.dispatch(
                acc_reg_comp_idx=st.acc_reg_idx,
                kernal_size=0 if layer.k_h == 3 else 1,
                line_buffer_row_shift=1,
                line_buffer_idx=st.line_buffer_idx,
                is_padding_col=1,
                weight_parall_mode=plan.weight_parall_mode,
                is_new=0,
                transnum=plan.weight_transnum_base,
                bas_addr=st.weight_bas_addr[0],
                is_bilinear_bicubic=plan.use_bilinear_weights,
                offset_reg_idx=st.offset_reg_idx,
            )
            st.acc_reg_idx = 1 - st.acc_reg_idx
            st.line_buffer_idx = 1 - st.line_buffer_idx

            isa.DataStorer.dispatch(
                quant_config_idx=st.quant_config_idx,
                pixelshuffle_out_mode=0,
                is_pixelshuffle=0,
                pooling_out_mode=0,
                pooling_out_new=0,
                is_pooling=0,
                reg_out_idx=st.acc_reg_idx,
                acc_mode=0,
                transfer_num=1,
                store_mode=0,
                stride=layer.h_in,
                base_addr_pooling=0,
                base_addrs_res=st.storer_bas_addr,
                is_bicubic_add=0,
                is_first_or_last_row=0,
                is_mask=0,
                is_new=0,
                dest_buffer_idx="a",
            )
            st.acc_reg_idx = 1 - st.acc_reg_idx
            st.storer_bas_addr += 2

    def _emit_deformable_conv(self, layer: LayerDesc, plan: TilingPlan) -> None:
        """
        Structural emission matching sd_sr_codegen (OffsetLoader before bilinear MAC).

        Address increments are simplified; schedules should be refined per layer.
        """
        st = self.state
        self.emit_quant_loader(layer.idx, transnum=max(4, layer.cout), quant_mode=0)
        cal_total = max(1, layer.h_in // plan.h_out_per_step)
        padding_num = plan.padding_num

        st.dataloader_bas_addr = 0
        st.storer_bas_addr = 0

        for cal_idx in range(cal_total):
            for ky in range(plan.ky_outer):
                isa.OffsetLoader.dispatch(
                    offset_reg_idx=st.offset_reg_idx,
                    bas_addr=cal_idx * plan.ky_outer + ky,
                )
                for ic_g in range(plan.ic_inner):
                    if cal_idx < padding_num and ky == 0:
                        is_padding_row = 4
                    elif cal_idx < padding_num and ky == 1:
                        is_padding_row = 1
                    elif cal_idx > cal_total - 1 - padding_num and ky == 1:
                        is_padding_row = 5
                    elif cal_idx > cal_total - 1 - padding_num and ky == 2:
                        is_padding_row = 6
                    else:
                        is_padding_row = 0

                    isa.DataLoader.dispatch(
                        layer_idx=layer.idx,
                        line_buffer_reshape=plan.line_buffer_reshape,
                        is_padding_row=is_padding_row,
                        read_mode=plan.read_mode,
                        transnum=plan.line_buffer_rows,
                        line_buffer_idx=st.line_buffer_idx,
                        src_buffer_idx="b",
                        bas_addr=st.dataloader_bas_addr + ic_g * 32 + (ky if cal_idx > 0 else 0),
                    )
                    st.line_buffer_idx = 1 - st.line_buffer_idx

                    isa.WeightLoader.dispatch(
                        acc_reg_comp_idx=st.acc_reg_idx,
                        kernal_size=0,
                        line_buffer_row_shift=5,
                        line_buffer_idx=st.line_buffer_idx,
                        is_padding_col=6,
                        weight_parall_mode=0,
                        is_new=0 if ky == 0 and ic_g == 0 else 1,
                        transnum=12,
                        bas_addr=st.weight_bas_addr[0] + 12 * (ky * plan.ic_inner + ic_g),
                        is_bilinear_bicubic=1,
                        offset_reg_idx=st.offset_reg_idx,
                    )
                    st.line_buffer_idx = 1 - st.line_buffer_idx

                st.offset_reg_idx = 1 - st.offset_reg_idx

            isa.DataStorer.dispatch(
                quant_config_idx=st.quant_config_idx,
                pixelshuffle_out_mode=0,
                is_pixelshuffle=0,
                pooling_out_mode=3,
                pooling_out_new=0,
                is_pooling=1,
                reg_out_idx=st.acc_reg_idx,
                acc_mode=4,
                transfer_num=1,
                store_mode=3,
                stride=32,
                base_addr_pooling=layer.h_in * 2,
                base_addrs_res=st.storer_bas_addr,
                is_bicubic_add=0,
                is_first_or_last_row=0,
                is_mask=0,
                is_new=0,
                dest_buffer_idx="a",
            )
            st.acc_reg_idx = 1 - st.acc_reg_idx
            st.storer_bas_addr += 4
            st.dataloader_bas_addr += 2 if cal_idx < padding_num else 4

        st.weight_bas_addr[0] += 12 * plan.ky_outer * plan.ic_inner


def emit_program(layers: List[LayerDesc], plans: List[TilingPlan]) -> List[Dict[str, Any]]:
    """Convenience: full network (caller orders layers)."""
    em = InstructionEmitter()
    em.reset()
    for L, P in zip(layers, plans):
        em.emit_layer(L, P)
    return list(isa.Inst.code_list)
