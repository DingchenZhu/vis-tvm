"""Stage-based compiler driver."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import tvm

from vis_compiler import relay_opt
from vis_compiler.emit.emitter import InstructionEmitter, emit_program
from vis_compiler.layer_desc import LayerDesc, extract_layer_descs
from vis_compiler.tiling import TilingPlan, plan_all


@dataclass
class PipelineConfig:
    dump_relay_path: Optional[str] = None
    dump_layers_path: Optional[str] = None
    run_optimize: bool = True


@dataclass
class StageResult:
    mod: tvm.ir.IRModule
    params: Dict[str, Any]
    layers: List[LayerDesc] = field(default_factory=list)
    tilings: List[TilingPlan] = field(default_factory=list)
    instructions: List[Dict[str, Any]] = field(default_factory=list)


class CompilerPipeline:
    """
    Composable stages: optimize → extract → tile → emit.

    Frontends should produce (mod, params); use `frontend.load_*` helpers.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self._hooks: Dict[str, List[Callable]] = {}

    def add_hook(self, stage: str, fn: Callable[[StageResult], None]) -> None:
        self._hooks.setdefault(stage, []).append(fn)

    def run(self, mod: tvm.ir.IRModule, params: Dict[str, Any]) -> StageResult:
        res = StageResult(mod=mod, params=params)

        if self.config.run_optimize:
            res.mod, res.params = relay_opt.optimize_for_codegen(res.mod, res.params)
        for h in self._hooks.get("after_opt", []):
            h(res)

        if self.config.dump_relay_path:
            from vis_compiler.frontend import dump_relay

            dump_relay(res.mod, self.config.dump_relay_path)

        res.layers = extract_layer_descs(res.mod)
        res.tilings = plan_all(res.layers)

        if self.config.dump_layers_path:
            import dataclasses
            import json

            payload = [
                {**dataclasses.asdict(L), "tiling": dataclasses.asdict(T)}
                for L, T in zip(res.layers, res.tilings)
            ]
            with open(self.config.dump_layers_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, default=str)

        for h in self._hooks.get("after_tile", []):
            h(res)

        res.instructions = emit_program(res.layers, res.tilings)

        for h in self._hooks.get("after_emit", []):
            h(res)

        return res
