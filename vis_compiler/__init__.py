"""
VIS accelerator compiler: Relay frontend → optimized IR → layer descriptions
→ tiling (per design guide) → micro-instruction emission (ISA wrappers).

This package is intentionally stage-based so new models/backends plug in
without copying monolithic codegen scripts.
"""

from vis_compiler.pipeline import CompilerPipeline, PipelineConfig, StageResult

__all__ = ["CompilerPipeline", "PipelineConfig", "StageResult"]
