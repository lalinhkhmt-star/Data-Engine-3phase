"""Lắp ráp SFT training set cuối cùng — bước "Final sampling" trong paper.

DDASPipeline.run() (layout) và .run_elements() (text/formula/table) chỉ CHỌN
mẫu (page id / Element) đã cân bằng diversity x difficulty — chưa phải data
huấn luyện thật. Module này tra pseudo-label đúng theo tier (label_source
trong cmcv.py) rồi đóng gói thành record, gộp cả 4 subtask lại một chỗ.

Easy/Medium -> nhãn tin cậy (đồng thuận model), dùng ngay được.
Hard/Invalid -> KHÔNG có nhãn tin cậy, tách sang hàng đợi riêng, không lẫn
vào tập train — phải qua Judge-and-Refine (§3.3, xem judge_refine.py) trước
khi dùng được, đúng cảnh báo "sẽ làm hỏng chứ không cải thiện model nếu dùng
trực tiếp" (dòng 45 paper).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .cmcv import CHEAP_EXTERNAL, TARGET_MODEL, ParseResult, Tier, label_source
from .element import Element


@dataclass
class SFTRecord:
    subtask: str                    # layout | text | formula | table
    page_id: str
    tier: Tier
    label_source: str               # model cung cấp nhãn (TARGET_MODEL hoặc CHEAP_EXTERNAL)
    content: object                 # layout: ParseResult của label_source; còn lại: str
    box: Optional[np.ndarray] = None    # None cho layout (áp dụng cả trang)
    element_id: Optional[int] = None


@dataclass
class HardItem:
    """Tham chiếu tới 1 sample Hard/Invalid — CHƯA có nhãn tin cậy, chờ Judge-and-Refine.

    `draft` là BẢN NHÁP để sửa, KHÔNG phải nhãn: Hard nghĩa là các model bất
    đồng, nhưng mỗi model vẫn có output riêng, và vòng Judge-and-Refine cần một
    điểm xuất phát để soi-và-sửa. Lấy output của TARGET_MODEL (model đang được
    cải thiện) làm nháp, thiếu thì lấy cheap external. None = không model nào có
    nội dung ở vùng này -> không có gì để sửa, đi thẳng sang người.
    """
    subtask: str
    page_id: str
    tier: Tier
    box: Optional[np.ndarray] = None
    element_id: Optional[int] = None
    draft: Optional[str] = None


def assemble_layout_records(
    final_page_ids: Sequence[int], tiers_by_page: Dict[int, Tier],
    parse_fn: Callable[[str], Tuple[ParseResult, ParseResult, Optional[ParseResult]]],
) -> Tuple[List[SFTRecord], List[HardItem]]:
    """`final_page_ids`/`tiers_by_page` từ `DDASPipeline.run()` — dùng đúng
    `cand_arr`/`tiers` để tra tier (KHÔNG suy luận model thêm, `parse_fn`
    phải trả lại ParseResult đã cache từ CMCV trang, giống Stage 2).
    """
    ready, hard = [], []
    for pid in final_page_ids:
        tier = tiers_by_page[int(pid)]
        src = label_source(tier)
        page_id = str(pid)
        if src is None:
            hard.append(HardItem("layout", page_id, tier))
            continue
        rm, rp, _rq = parse_fn(page_id)
        chosen = rm if src == rm.model else rp
        ready.append(SFTRecord("layout", page_id, tier, src, chosen))
    return ready, hard


def assemble_element_records(
    elements_by_subtask: Dict[str, List[Element]],
) -> Tuple[List[SFTRecord], List[HardItem]]:
    ready, hard = [], []
    for subtask, elements in elements_by_subtask.items():
        for el in elements:
            src = label_source(el.tier)
            if src is None or src not in el.content:
                # Hard, hoặc Easy/Medium nhưng thiếu đúng model nguồn nhãn
                # trong content (không nên xảy ra theo logic derive_element_cmcv,
                # nhưng không giả định — không đủ bằng chứng thì không đoán).
                draft = el.content.get(TARGET_MODEL) or el.content.get(CHEAP_EXTERNAL)
                hard.append(HardItem(subtask, el.page_id, el.tier, el.box, el.eid, draft))
                continue
            ready.append(SFTRecord(subtask, el.page_id, el.tier, src,
                                   el.content[src], el.box, el.eid))
    return ready, hard


def assemble_sft_set(
    final_layout_page_ids: Sequence[int], tiers_by_page: Dict[int, Tier],
    parse_fn: Callable[[str], Tuple[ParseResult, ParseResult, Optional[ParseResult]]],
    elements_by_subtask: Dict[str, List[Element]],
) -> Tuple[List[SFTRecord], List[HardItem]]:
    """Gộp cả 4 subtask (layout + text/formula/table) thành 1 SFT set duy nhất.

    Trả về (sft_ready, hard_queue): sft_ready dùng train ngay (Easy+Medium,
    đã cân bằng diversity x difficulty ở bước chọn mẫu trước đó); hard_queue
    là tham chiếu Hard/Invalid CHƯA có nhãn, chuyển tiếp sang Judge-and-Refine.
    """
    layout_ready, layout_hard = assemble_layout_records(final_layout_page_ids, tiers_by_page, parse_fn)
    elem_ready, elem_hard = assemble_element_records(elements_by_subtask)
    return layout_ready + elem_ready, layout_hard + elem_hard


def summarize(sft_ready: List[SFTRecord], hard_queue: List[HardItem]) -> Dict[str, Dict[str, int]]:
    """Đếm theo subtask — dùng để log/kiểm tra độ phủ 4 subtask trước khi xuất file."""
    out: Dict[str, Dict[str, int]] = {}
    for st in ("layout", "text", "formula", "table"):
        ready_n = sum(1 for r in sft_ready if r.subtask == st)
        hard_n = sum(1 for h in hard_queue if h.subtask == st)
        out[st] = {"sft_ready": ready_n, "hard_queue": hard_n}
    return out


# ------------------------------------------- phân tầng theo giai đoạn train --
#
# Paper dòng 66: "65.5M Easy and Medium samples ... Stage 1 pre-training; 192K
# expert-annotated Hard samples are used for Stage 2 fine-tuning AND Stage 3
# GRPO alignment" — nói cả hai giai đoạn dùng chung tập 192K nhưng KHÔNG cho
# tiêu chí chia. Tiêu chí dưới đây là tự đề xuất, suy từ nhu cầu khác nhau của
# hai thuật toán:
#
#   SFT (Stage 2) cần cặp (đầu vào -> MỘT đáp án đúng). Mẫu nào cũng dùng được.
#   GRPO (Stage 3) học từ PHẦN THƯỞNG so giữa nhiều đáp án sinh ra. Muốn có
#   gradient thì cần (a) chấm được đúng/sai TỰ ĐỘNG lúc train, và (b) model
#   hiện còn sai thật để có cái mà so.
#
# => Mẫu vào GRPO phải KIỂM CHỨNG ĐƯỢC BẰNG MÁY: formula (compile + render so
# ảnh) và table (dựng lại lưới ô so với nhãn) chấm tự động được; text và layout
# thì không — muốn biết đúng sai phải có người, không dùng làm reward được.
# Trong nhóm kiểm chứng được, ưu tiên mẫu CMCV bất đồng mạnh (model đang sai
# nhiều nhất ở đó => reward có tín hiệu).
#
# Mẫu không vào GRPO thì về Stage 2. Không mẫu nào bị bỏ phí.

GRPO_VERIFIABLE = ("formula", "table")


def split_training_stages(expert_records: Sequence[SFTRecord],
                          weakness: Optional[Dict[str, float]] = None,
                          grpo_ratio: float = 0.30,
                          ) -> Tuple[List[SFTRecord], List[SFTRecord]]:
    """Chia tập Hard đã chú thích tay thành (stage2_sft, stage3_grpo).

    `grpo_ratio` = trần tỉ lệ dành cho GRPO (mặc định 30% — GRPO cần ít dữ liệu
    hơn SFT nhiều, và mẫu ở đây quá đắt để dồn hết vào một giai đoạn). CHƯA
    hiệu chỉnh: con số đúng chỉ biết được sau khi chạy thử cả hai giai đoạn.
    `weakness` (từ judge_refine.weakness_by_subtask) để xếp trong nhóm kiểm
    chứng được, subtask nào model yếu nhất thì vào GRPO trước.
    """
    w = weakness or {}
    verifiable = [r for r in expert_records if r.subtask in GRPO_VERIFIABLE]
    rest = [r for r in expert_records if r.subtask not in GRPO_VERIFIABLE]

    verifiable.sort(key=lambda r: -w.get(r.subtask, 0.0))
    cap = int(len(expert_records) * grpo_ratio)
    grpo = verifiable[:cap]
    sft = verifiable[cap:] + rest
    return sft, grpo
