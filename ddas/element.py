"""Stage 2 — Lấy mẫu mức element, CMCV element *dẫn xuất* từ CMCV trang.

Chi tiết hiệu quả quan trọng: không chạy thêm lượt suy luận VLM nào cho element.
Đầu ra trang của ba model đã chứa toàn bộ element (bbox + nội dung). Chỉ cần
căn chỉnh bbox giữa các model (Hungarian theo IoU) rồi áp lại đúng độ đo của
subtask lên từng element đã ghép.
=> chi phí element-level CMCV là CPU thuần, ~0 GPU-hour, thay vì nhân 3 lần
   suy luận trên ~1.8B element.

Element không được ghép (chỉ 1 model phát hiện) là tín hiệu Hard cho subtask
layout — chính là chỗ các model bất đồng về chính việc "có gì trên trang".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from .cmcv import CMCVRecord, ParseResult, Tier, assign_tier
from .config import CMCVConfig
from .metrics import SIM_FN, iou_matrix


@dataclass
class Element:
    page_id: str
    eid: int
    etype: str                 # 'text' | 'formula' | 'table'
    box: np.ndarray            # xyxy trên trang gốc
    content: Dict[str, str]    # model -> nội dung
    tier: Tier
    sims: Dict[str, Optional[float]]


def _elements_of(r: ParseResult) -> List[Tuple[str, np.ndarray, str]]:
    """Làm phẳng ParseResult thành (etype, box, content), khớp theo thứ tự đọc."""
    out, fi, ti = [], 0, 0
    for box, lab in zip(r.boxes, r.labels):
        if lab == "formula":
            out.append(("formula", box, r.formulas[fi] if fi < len(r.formulas) else "")); fi += 1
        elif lab == "table":
            out.append(("table", box, r.tables[ti] if ti < len(r.tables) else "")); ti += 1
        elif lab in ("text", "title", "list", "caption"):
            out.append(("text", box, ""))
    return out


def match_elements(ra: ParseResult, rb: ParseResult, iou_thr: float = 0.5):
    """Ghép 1-1 tối ưu giữa element của hai model (Hungarian trên -IoU)."""
    ea, eb = _elements_of(ra), _elements_of(rb)
    if not ea or not eb:
        return [], ea, eb
    M = iou_matrix(np.stack([e[1] for e in ea]), np.stack([e[1] for e in eb]))
    same = np.array([[a[0] == b[0] for b in eb] for a in ea])
    M = np.where(same, M, 0.0)
    ri, ci = linear_sum_assignment(-M)
    pairs = [(int(i), int(j)) for i, j in zip(ri, ci) if M[i, j] >= iou_thr]
    mi, mj = {i for i, _ in pairs}, {j for _, j in pairs}
    return pairs, [ea[i] for i in range(len(ea)) if i not in mi], \
                  [eb[j] for j in range(len(eb)) if j not in mj]


def derive_element_cmcv(page_id: str,
                        rm: ParseResult, rp: ParseResult, rq: Optional[ParseResult],
                        cfg: CMCVConfig | None = None) -> List[Element]:
    """CMCV mức element, tái sử dụng đúng taxonomy Easy/Medium/Hard của trang."""
    cfg = cfg or CMCVConfig()
    pairs_mp, _, _ = match_elements(rm, rp)
    em, ep = _elements_of(rm), _elements_of(rp)
    eq = _elements_of(rq) if rq is not None else []
    idx_mq = {i: j for i, j in (match_elements(rm, rq)[0] if rq is not None else [])}
    idx_pq = {i: j for i, j in (match_elements(rp, rq)[0] if rq is not None else [])}

    out: List[Element] = []
    for k, (i, j) in enumerate(pairs_mp):
        etype, box, ca = em[i]
        cb = ep[j][2]
        fn = SIM_FN[etype] if etype in SIM_FN else SIM_FN["text"]
        tau = cfg.tau[etype]
        s_mp = fn(ca, cb) if etype != "text" else 1.0 if not (ca or cb) else fn(ca, cb)
        s_mq = s_pq = None
        if rq is not None:
            if i in idx_mq:
                s_mq = fn(ca, eq[idx_mq[i]][2])
            if j in idx_pq:
                s_pq = fn(cb, eq[idx_pq[j]][2])
        tier = assign_tier(s_mp, s_mq, s_pq, tau, cfg.require_3way_for_easy)
        out.append(Element(page_id, k, etype, box, {"mineru": ca, "paddle": cb},
                           tier, {"M-P": s_mp, "M-Q": s_mq, "P-Q": s_pq}))
    return out


# --------------------------------------------------- đặc trưng element ------

def element_feature(crop_embed: np.ndarray, box: np.ndarray,
                    page_wh: Tuple[float, float], etype: str,
                    extra: Optional[np.ndarray] = None) -> np.ndarray:
    """Đặc trưng cụm mức element: hình ảnh crop + tiên nghiệm hình học.

    Tiên nghiệm hình học (tỉ lệ khung, diện tích tương đối, vị trí) rẻ nhưng
    tách rất tốt: bảng rộng-thấp vs bảng lồng nhau, công thức inline vs display.
    """
    w, h = page_wh
    x1, y1, x2, y2 = box
    bw, bh = max(x2 - x1, 1e-6), max(y2 - y1, 1e-6)
    geo = np.array([bw / w, bh / h, (bw * bh) / (w * h), bw / bh,
                    (x1 + x2) / (2 * w), (y1 + y2) / (2 * h)], np.float32)
    parts = [crop_embed.astype(np.float32), geo]
    if extra is not None:
        parts.append(extra.astype(np.float32))
    v = np.concatenate(parts)
    return v / max(np.linalg.norm(v), 1e-9)
