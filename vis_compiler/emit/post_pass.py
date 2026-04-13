"""
Post-process raw ISA dicts: field alignment, dependency edges, virtual dest/src registers.

Ported from references/sd_sr_codegen.py __main__ (dependency + register allocation).
Dependency list entries are instruction indices into ``code_list``; register resolution
matches the reference by finding the producer whose ``code_num`` contains that index
(when ``code_num[i] == [i]`` for all i, this is equivalent to indexing by id).
"""
from __future__ import annotations

import ast
import copy
from pathlib import Path
from typing import Any, Dict, List


def align_instruction_fields(code_list: List[Dict[str, Any]]) -> None:
    """Add golden / encoder fields missing from minimal isa.dispatch records."""
    for c in code_list:
        op = c.get("op_code")
        if op == "OffchipDataLoader":
            c.setdefault("is_compression", 0)
        elif op == "DataLoader":
            c.setdefault("offchip_read_mode", 0)
            c.setdefault("is_compression", 0)
        elif op == "WeightLoader":
            c.setdefault("is_skip", 2)
            c.setdefault("is_new", 1)
        elif op == "DataStorer":
            c.setdefault("is_offset", 0)
        elif op == "OffchipDataStorer":
            c.setdefault("is_compression", 0)


def add_instruction_dependencies(code_list: List[Dict[str, Any]]) -> int:
    """
    Fill ``dependency`` with producer instruction indices (sd_sr_codegen rules).
    Returns max_gap (largest i - dep_index).
    """
    n = len(code_list)
    for i in range(n):
        code_list[i]["dependency"] = []
        code_list[i]["dest"] = 0
        code_list[i]["src1"] = 0
        code_list[i]["src2"] = 0
        code_list[i]["src3"] = 0
        code_list[i]["src4"] = 0

        if code_list[i]["op_code"] == "OffchipDataLoader":
            for d in range(i - 1, -1, -1):
                if code_list[d]["op_code"] == "DataLoader":
                    if code_list[d]["layer_idx"] == 0:
                        code_list[i]["dependency"].append(d)
                        break
            for d in range(i - 1, -1, -1):
                if code_list[d]["op_code"] == "OffchipDataLoader":
                    code_list[i]["dependency"].append(d)
                    break
        elif code_list[i]["op_code"] == "WeightLoader":
            for d in range(i - 1, -1, -1):
                if code_list[d]["op_code"] == "DataLoader":
                    if code_list[d]["line_buffer_idx"] == code_list[i]["line_buffer_idx"]:
                        code_list[i]["dependency"].append(d)
                        break
            for d in range(i - 1, -1, -1):
                if code_list[d]["op_code"] == "DataStorer":
                    if code_list[d]["reg_out_idx"] == code_list[i]["acc_reg_comp_idx"]:
                        code_list[i]["dependency"].append(d)
                        break
                if code_list[d]["op_code"] == "WeightLoader":
                    if code_list[d]["acc_reg_comp_idx"] == code_list[i]["acc_reg_comp_idx"]:
                        break
            for d in range(i - 1, -1, -1):
                if code_list[d]["op_code"] == "WeightLoader":
                    code_list[i]["dependency"].append(d)
                    break
            if code_list[i].get("is_bilinear_bicubic") == 1 and code_list[i]["dependency"]:
                last = code_list[i]["dependency"][-1]
                last_w = code_list[last]
                if last_w["is_bilinear_bicubic"] == 0 or (
                    last_w["is_bilinear_bicubic"] == 1
                    and last_w["offset_reg_idx"] != code_list[i]["offset_reg_idx"]
                ):
                    for d in range(i - 1, -1, -1):
                        if code_list[d]["op_code"] == "OffsetLoader":
                            if code_list[d]["offset_reg_idx"] == code_list[i]["offset_reg_idx"]:
                                code_list[i]["dependency"].append(d)
                                break
        elif code_list[i]["op_code"] == "DataLoader":
            for d in range(i - 1, -1, -1):
                if code_list[d]["op_code"] == "WeightLoader":
                    if code_list[d]["line_buffer_idx"] == code_list[i]["line_buffer_idx"]:
                        code_list[i]["dependency"].append(d)
                        break
            if code_list[i]["layer_idx"] == 0:
                dataloader_count = 0
                for d in range(i - 1, -1, -1):
                    if code_list[d]["op_code"] == "DataLoader" and dataloader_count < 2:
                        if code_list[d]["layer_idx"] == 0:
                            dataloader_count += 1
                    if dataloader_count == 2 or code_list[d]["op_code"] == "OffchipDataStorer":
                        break
                if dataloader_count < 2:
                    if code_list[i]["bas_addr"] < 144 * 4:
                        for d in range(i - 1, -1, -1):
                            if code_list[d]["op_code"] == "OffchipDataLoader":
                                if (
                                    code_list[d]["src_buffer_idx"] == 0
                                    and code_list[d]["load_model"] == 0
                                ):
                                    code_list[i]["dependency"].append(d)
                                    break
                    if code_list[i]["bas_addr"] >= 144 * 4:
                        for d in range(i - 1, -1, -1):
                            if code_list[d]["op_code"] == "OffchipDataLoader":
                                if (
                                    code_list[d]["src_buffer_idx"] == 0
                                    and code_list[d]["load_model"] == 1
                                ):
                                    code_list[i]["dependency"].append(d)
                                    break
            else:
                dataloader_count = 0
                dataloader_idx: List[int] = []
                datastorer_count = 0
                datastorer_idx: List[int] = []
                for d in range(i - 1, -1, -1):
                    if code_list[d]["op_code"] == "DataLoader" and dataloader_count < 2:
                        dataloader_count += 1
                        dataloader_idx.append(d)
                    if code_list[d]["op_code"] == "DataStorer" and datastorer_count < 1:
                        datastorer_count += 1
                        datastorer_idx.append(d)
                    if dataloader_count == 2 and datastorer_count == 1:
                        break
                if (
                    len(dataloader_idx) >= 2
                    and len(datastorer_idx) >= 1
                    and (
                        code_list[dataloader_idx[0]]["layer_idx"] != code_list[i]["layer_idx"]
                        or code_list[dataloader_idx[1]]["layer_idx"] != code_list[i]["layer_idx"]
                    )
                ):
                    code_list[i]["dependency"].append(datastorer_idx[0])
        elif code_list[i]["op_code"] == "QuantLoader":
            for d in range(i - 1, -1, -1):
                if code_list[d]["op_code"] == "DataStorer":
                    if code_list[d]["quant_config_idx"] == code_list[i]["quant_reg_load_idx"]:
                        code_list[i]["dependency"].append(d)
                        break
            quantloader_count = 0
            for d in range(i - 1, -1, -1):
                if code_list[d]["op_code"] == "OffchipDataLoader":
                    if code_list[d]["src_buffer_idx"] == 2:
                        code_list[i]["dependency"].append(d)
                        break
                if code_list[d]["op_code"] == "QuantLoader":
                    quantloader_count += 1
                if quantloader_count == 2:
                    break
        elif code_list[i]["op_code"] == "DataStorer":
            for d in range(i - 1, -1, -1):
                if code_list[d]["op_code"] == "QuantLoader":
                    if code_list[d]["quant_reg_load_idx"] == code_list[i]["quant_config_idx"]:
                        code_list[i]["dependency"].append(d)
                        break
                if code_list[d]["op_code"] == "DataStorer":
                    if code_list[d]["quant_config_idx"] == code_list[i]["quant_config_idx"]:
                        break
            for d in range(i - 1, -1, -1):
                if code_list[d]["op_code"] == "WeightLoader":
                    if code_list[d]["acc_reg_comp_idx"] == code_list[i]["reg_out_idx"]:
                        code_list[i]["dependency"].append(d)
                        break
            for d in range(i - 1, -1, -1):
                if code_list[d]["op_code"] == "DataStorer":
                    code_list[i]["dependency"].append(d)
                    break
            if code_list[i]["dest_buffer_idx"] in ("fsrcnn_output_buffer", "unet_output_reg"):
                for d in range(i - 1, -1, -1):
                    if code_list[d]["op_code"] == "OffchipDataStorer":
                        code_list[i]["dependency"].append(d)
                        break
        elif code_list[i]["op_code"] == "OffsetLoader":
            for d in range(i - 1, -1, -1):
                if code_list[d]["op_code"] == "WeightLoader":
                    if (
                        code_list[d]["offset_reg_idx"] == code_list[i]["offset_reg_idx"]
                        and code_list[d]["is_bilinear_bicubic"] == 1
                    ):
                        code_list[i]["dependency"].append(d)
                        break
                    if code_list[d]["is_bilinear_bicubic"] == 0:
                        break
            offsetloader_count = 0
            for d in range(i - 1, -1, -1):
                if code_list[d]["op_code"] == "OffsetLoader":
                    offsetloader_count += 1
                    if offsetloader_count == 2:
                        break
                if code_list[d]["op_code"] == "DataStorer":
                    if code_list[d]["dest_buffer_idx"] == "offset_reg":
                        code_list[i]["dependency"].append(d)
                        break
        elif code_list[i]["op_code"] == "OffchipDataStorer":
            for d in range(i - 1, -1, -1):
                if code_list[d]["op_code"] == "DataStorer":
                    if code_list[d]["dest_buffer_idx"] == code_list[i]["src_buffer"]:
                        code_list[i]["dependency"].append(d)
                        break

    max_gap = 0
    for i in range(n):
        for t in code_list[i]["dependency"]:
            if i - t > max_gap:
                max_gap = i - t
    return max_gap


def assign_dependency_registers(
    code_list: List[Dict[str, Any]], max_gap: int
) -> int:
    """
    Assign dest and src1..src4 from dependency graph (sd_sr_codegen register pool 1..15).
    Returns peak live virtual register count.
    """
    idle_reg_id = list(range(1, 16))[::-1]
    init_len = len(idle_reg_id)
    reg_used_count_max = 0
    occupy_list: List[List[int]] = []

    for i, code_dict in enumerate(code_list):
        dest_code: List[int] = []
        src_code: List[int] = []
        for one_code_num in code_dict["code_num"]:
            reg_id = idle_reg_id.pop()
            occupy_list.append([one_code_num, reg_id])
            dest_code.append(reg_id)
        assert 0 < len(dest_code) <= 2
        code_dict["dest"] = dest_code[0]

        reg_used_count = init_len - len(idle_reg_id)
        reg_used_count_max = max(reg_used_count_max, reg_used_count)

        for dep_ref in code_dict["dependency"]:
            for j in range(i - 1, -1, -1):
                if dep_ref in code_list[j]["code_num"]:
                    src_code.append(code_list[j]["dest"])
                    break

        assert len(src_code) <= 4
        code_dict["src1"] = src_code[0] if len(src_code) > 0 else 0
        code_dict["src2"] = src_code[1] if len(src_code) > 1 else 0
        code_dict["src3"] = src_code[2] if len(src_code) > 2 else 0
        # Match reference sd_sr_codegen (same quirk as legacy script)
        code_dict["src4"] = src_code[2] if len(src_code) > 3 else 0

        for occ_pair in reversed(occupy_list):
            required = False
            upper = min(len(code_list), i + max_gap + 10)
            for k in range(i + 1, upper):
                if occ_pair[0] in code_list[k]["dependency"]:
                    required = True
                    break
            if not required:
                idle_reg_id.append(occ_pair[1])
                occupy_list.remove(occ_pair)

    return reg_used_count_max


def finalize_instructions(
    code_list: List[Dict[str, Any]],
    *,
    align_fields: bool = True,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Mutates ``code_list`` in place. Returns simple stats for logging / tests.
    """
    if align_fields:
        align_instruction_fields(code_list)
    max_gap = add_instruction_dependencies(code_list)
    reg_max = assign_dependency_registers(code_list, max_gap)
    stats = {"max_gap": max_gap, "reg_used_count_max": reg_max, "num_instructions": len(code_list)}
    if verbose:
        print(stats)
    return stats


def strip_post_pass_fields(code_list: List[Dict[str, Any]]) -> None:
    """Remove fields produced by finalize (for golden replay tests)."""
    for c in code_list:
        for k in ("dependency", "dest", "src1", "src2", "src3", "src4"):
            c.pop(k, None)


def prepend_leading_code_num_padding(code_list: List[Dict[str, Any]]) -> int:
    """
    Golden text dumps sometimes start at ``code_num`` > 0 (earlier ops omitted).
    Prepend stub rows ``code_num`` ``[0]`` … ``[n-1]`` with ``op_code`` ``PaddingNoOp``
    so dependency indices that refer to those ids resolve. Returns ``n`` (rows
    prepended), or ``0`` if none.
    """
    if not code_list:
        return 0
    nums = code_list[0].get("code_num") or [0]
    n0 = int(nums[0]) if nums else 0
    if n0 <= 0:
        return 0
    stubs: List[Dict[str, Any]] = [
        {"code_num": [k], "op_code": "PaddingNoOp"} for k in range(n0)
    ]
    code_list[:] = stubs + code_list
    return n0


def load_golden_pseudo_code_file(path: str | Path) -> List[Dict[str, Any]]:
    """Load one instruction dict per line (``ast.literal_eval``), same format as ``golden/*.txt``."""
    text = Path(path).read_text(encoding="utf-8")
    out: List[Dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(ast.literal_eval(line))
    return out


def instruction_indices_aligned_with_code_num(code_list: List[Dict[str, Any]]) -> bool:
    """True if ``code_list[i]['code_num'][0] == i`` for every row (fresh ``isa`` emit)."""
    for i, c in enumerate(code_list):
        nums = c.get("code_num") or []
        if not nums or int(nums[0]) != i:
            return False
    return True


def verify_instruction_list_roundtrip(code_list: List[Dict[str, Any]]) -> bool:
    """
    Strip post-pass fields and re-run ``finalize_instructions``; result must equal
    the input. Only meaningful when :func:`instruction_indices_aligned_with_code_num`
    is true (compiler output). Repository ``golden/*.txt`` slices usually **fail** this
    because they omit a prefix and/or use a different ``code_num`` base.
    """
    if not instruction_indices_aligned_with_code_num(code_list):
        return False
    work = copy.deepcopy(code_list)
    strip_post_pass_fields(work)
    finalize_instructions(work)
    return work == code_list


def verify_golden_post_pass(path: str | Path) -> bool | None:
    """
    If the file starts at ``code_num`` ``[0]``, verify round-trip strip + finalize.
    Otherwise return ``None`` (cannot validate sliced / renumbered dumps).
    """
    golden = load_golden_pseudo_code_file(path)
    if not golden:
        return True
    if golden[0].get("code_num", [None])[0] != 0:
        return None
    return verify_instruction_list_roundtrip(golden)
