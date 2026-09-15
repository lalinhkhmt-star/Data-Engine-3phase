"""CMCV — Cross-Model Consistency Verification (Section 3.2) với cascade.

Điểm cốt lõi so với bản trong paper: thứ tự đánh giá được sắp xếp lại thành
cascade *không đổi nhãn* (lossless). Vì Easy chỉ cần MinerU đồng thuận với
ÍT NHẤT MỘT external model, nên khi MinerU ~ PaddleOCR-VL (hai model <1.5B, rẻ)
ta đã kết luận được Easy mà không cần gọi Qwen3-VL-30B.
=> tiết kiệm ~60% lời gọi model đắt nhất, nhãn cuối hoàn toàn giống hệt.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .config import CMCVConfig
from .metrics import SIM_FN, layout_sim

MINERU, PADDLE, QWEN = "mineru2.5", "paddleocr-vl", "qwen3-vl-30b"
TARGET_MODEL = MINERU                      # model đang được cải thiện
EXTERNALS = (PADDLE, QWEN)
CHEAP_EXTERNAL = PADDLE                    # ~0.9B, chạy được trên toàn pool
EXPENSIVE_EXTERNAL = QWEN                  # ~30B MoE, chỉ gọi khi cascade không cắt được


class Tier(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    INVALID = "invalid"


@dataclass
class ParseResult:
    """Đầu ra chuẩn hoá của một model trên một trang."""
    page_id: str
    model: str
    text: str = ""
    boxes: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), np.float32))
    labels: List[str] = field(default_factory=list)
    formulas: List[str] = field(default_factory=list)   # LaTeX, theo thứ tự đọc
    tables: List[str] = field(default_factory=list)     # HTML, theo thứ tự đọc
    latency_ms: float = 0.0


@dataclass
class CMCVRecord:
    page_id: str
    tier: Dict[str, Tier]                  # tier cho từng subtask
    sims: Dict[str, Dict[str, float]]      # subtask -> {"M-P":.., "M-Q":.., "P-Q":..}
    used_expensive: bool
    pseudo_label_from: Dict[str, Optional[str]]   # subtask -> model cung cấp nhãn


# ------------------------------------------------------------ similarity ----

def _seq_sim(a: Sequence[str], b: Sequence[str], fn: Callable[[str, str], float]) -> float:
    """So khớp hai danh sách (formula/table) theo thứ tự đọc, phạt lệch số lượng."""
    if not a and not b:
        return 1.0
    n = max(len(a), len(b))
    m = min(len(a), len(b))
    return sum(fn(x, y) for x, y in zip(a[:m], b[:m])) / n


def pair_sims(r1: ParseResult, r2: ParseResult) -> Dict[str, float]:
    return {
        "text": SIM_FN["text"](r1.text, r2.text),
        "formula": _seq_sim(r1.formulas, r2.formulas, SIM_FN["formula"]),
        "table": _seq_sim(r1.tables, r2.tables, SIM_FN["table"]),
        "layout": layout_sim(r1.boxes, r1.labels, r2.boxes, r2.labels),
    }


# ------------------------------------------------------------- tiering ------

def assign_tier(s_mp: float, s_mq: Optional[float], s_pq: Optional[float],
                tau: float, require_3way: bool = False) -> Tier:
    """Phân tầng theo đúng taxonomy Section 3.2, neo vào MinerU2.5.

    s_mq / s_pq = None nghĩa là cascade đã cắt sớm (chưa chạy model đắt).
    """
    if s_mq is None:                       # cascade đã cắt => chắc chắn Easy
        return Tier.EASY
    agree_mp, agree_mq = s_mp >= tau, s_mq >= tau
    if require_3way:
        if agree_mp and agree_mq:
            return Tier.EASY
    elif agree_mp or agree_mq:
        return Tier.EASY
    if s_pq is not None and s_pq >= tau:
        return Tier.MEDIUM                 # hai external đồng thuận, MinerU lệch
    return Tier.HARD


def _label_source(tier: Tier) -> Optional[str]:
    return {Tier.EASY: MINERU, Tier.MEDIUM: PADDLE, Tier.HARD: None,
            Tier.INVALID: None}[tier]


# ------------------------------------------------------------- cascade ------

class CMCV:
    """Chạy CMCV trên một trang. `runners` là dict model_name -> callable(page)->ParseResult.

    Cascade tiết kiệm: gọi MinerU + PaddleOCR trước; chỉ gọi Qwen3-VL-30B khi
    tồn tại subtask mà MinerU và PaddleOCR bất đồng.
    """

    def __init__(self, runners: Dict[str, Callable[[str], ParseResult]],
                 cfg: CMCVConfig | None = None,
                 validity_fn: Callable[[str], bool] | None = None):
        self.runners = runners
        self.cfg = cfg or CMCVConfig()
        self.validity_fn = validity_fn
        self.stats = {"pages": 0, "expensive_calls": 0}

    def run_page(self, page_id: str) -> CMCVRecord:
        self.stats["pages"] += 1
        subtasks = tuple(self.cfg.tau.keys())

        if self.validity_fn is not None and not self.validity_fn(page_id):
            return CMCVRecord(page_id, {t: Tier.INVALID for t in subtasks},
                              {t: {} for t in subtasks}, False,
                              {t: None for t in subtasks})

        rm = self.runners[MINERU](page_id)
        rp = self.runners[CHEAP_EXTERNAL](page_id)
        s_mp = pair_sims(rm, rp)

        # Subtask nào MinerU ~ Paddle thì đã là Easy -> không cần model đắt.
        need_q = [t for t in subtasks
                  if s_mp[t] < self.cfg.tau[t] or self.cfg.require_3way_for_easy]
        run_q = (not self.cfg.cascade) or bool(need_q)
        rq = None
        if run_q:
            rq = self.runners[EXPENSIVE_EXTERNAL](page_id)
            self.stats["expensive_calls"] += 1
        s_mq = pair_sims(rm, rq) if rq is not None else None
        s_pq = pair_sims(rp, rq) if rq is not None else None

        tiers, sims, src = {}, {}, {}
        for t in subtasks:
            mq = s_mq[t] if s_mq is not None else None
            pq = s_pq[t] if s_pq is not None else None
            tier = assign_tier(s_mp[t], mq, pq, self.cfg.tau[t],
                               self.cfg.require_3way_for_easy)
            tiers[t] = tier
            sims[t] = {"M-P": s_mp[t], "M-Q": mq, "P-Q": pq}
            src[t] = _label_source(tier)
        return CMCVRecord(page_id, tiers, sims, rq is not None, src)

    @property
    def expensive_call_rate(self) -> float:
        return self.stats["expensive_calls"] / max(1, self.stats["pages"])


# ---------------------------------------------------------- calibration -----

def calibrate_tau(sims: np.ndarray, correct: np.ndarray,
                  precision_target: float = 0.98,
                  grid: np.ndarray | None = None) -> Dict[str, float]:
    """Chọn tau nhỏ nhất sao cho P(đúng | sim >= tau) >= precision_target.

    `sims`   : điểm đồng thuận của cặp model trên dev-set có ground truth.
    `correct`: đầu ra của model có khớp GT hay không (bool).
    Đây là bước BẮT BUỘC: giả định "đồng thuận => đúng" phải được đo, không được
    mặc định — nhất là khi hai model có thể mắc lỗi tương quan.
    """
    grid = np.linspace(0.5, 0.999, 200) if grid is None else grid
    best = {"tau": 1.0, "precision": 1.0, "coverage": 0.0}
    for t in grid:
        m = sims >= t
        if m.sum() < 30:
            continue
        p = float(correct[m].mean())
        if p >= precision_target:
            best = {"tau": float(t), "precision": p, "coverage": float(m.mean())}
            break
    return best
