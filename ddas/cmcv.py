"""CMCV — Cross-Model Consistency Verification (Section 3.2) với cascade.

Bộ 3 model chọn riêng cho tiếng Việt (thay cho MinerU2.5 gốc — MinerU huấn
luyện chủ yếu trên corpus tiếng Trung/Anh, không phải mục tiêu tối ưu ở đây):
  - TARGET_MODEL (Qwen3-VL, TỰ HOST) là model đang được cải thiện — output
    của nó trên trang Easy trở thành nhãn SFT, nên phải là model bạn thực sự
    định tiếp tục fine-tune. Hai model dưới đây gọi qua API, KHÔNG fine-tune
    được (closed-weight) nên không thể làm target.
  - CHEAP_EXTERNAL (Mistral OCR, qua API) — $4/1000 trang, output có sẵn
    bbox + 13 loại block + bảng/công thức dạng structured JSON, khớp gần
    đúng format ParseResult cần, không phải tự parse text thô.
  - EXPENSIVE_EXTERNAL (Gemini 3 Pro, qua API) — trọng tài cho case Hard,
    chỉ gọi khi cascade không cắt được nên tần suất thấp, chấp nhận giá cao
    hơn ($2/$12 mỗi triệu token in/out). LƯU Ý: Gemini 2.5 Pro đã bị Google
    khai tử (16/10/2026) — đừng dùng nhầm bản cũ.
  - Cả hai external đều khác lineage với target (Mistral Pháp, Google Mỹ,
    Qwen Alibaba Trung Quốc) nên "đồng thuận" có ý nghĩa thật — nhưng vẫn
    PHẢI đo lại trên dev-set tiếng Việt bằng calibrate_tau() trước khi tin,
    đừng mặc định giả định "đồng thuận ⇒ đúng". Nếu nghi lỗi tương quan giữa
    2 external, dùng key GPT-4o/GPT-5 (ChatGPT) làm model thứ 3 kiểm tra.

Điểm cốt lõi so với bản trong paper: thứ tự đánh giá được sắp xếp lại thành
cascade *không đổi nhãn* (lossless). Vì Easy chỉ cần target đồng thuận với
ÍT NHẤT MỘT external model, nên khi target ~ Mistral OCR (rẻ) ta đã kết luận
được Easy mà không cần gọi Gemini 3 Pro (đắt hơn nhiều lần trên mỗi trang).
=> tiết kiệm phần lớn lời gọi model đắt nhất, nhãn cuối hoàn toàn giống hệt.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .config import CMCVConfig
from .metrics import SIM_FN, layout_sim

TARGET_MODEL = "qwen3-vl-8b"                # model đang được cải thiện — TỰ HOST
CHEAP_EXTERNAL = "mistral-ocr-4"            # qua API, ~$4/1000 trang, chạy được trên toàn pool
EXPENSIVE_EXTERNAL = "gemini-3-pro"         # qua API, trọng tài, chỉ gọi khi cascade không cắt được
EXTERNALS = (CHEAP_EXTERNAL, EXPENSIVE_EXTERNAL)


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
    """Phân tầng theo đúng taxonomy Section 3.2, neo vào TARGET_MODEL.

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
        return Tier.MEDIUM                 # hai external đồng thuận, target lệch
    return Tier.HARD


def label_source(tier: Tier) -> Optional[str]:
    """Model nào cung cấp pseudo-label cho tier này. None = chưa có nhãn tin
    cậy (Hard/Invalid) — dùng khi lắp ráp SFT set, xem sft.py."""
    return {Tier.EASY: TARGET_MODEL, Tier.MEDIUM: CHEAP_EXTERNAL, Tier.HARD: None,
            Tier.INVALID: None}[tier]


# ------------------------------------------------------------- cascade ------

class CMCV:
    """Chạy CMCV trên một trang. `runners` là dict model_name -> callable(page)->ParseResult.

    Cascade tiết kiệm: gọi target (Qwen3-VL, tự host) + Mistral OCR (API) trước;
    chỉ gọi model đắt (Gemini 3 Pro, API) khi tồn tại subtask mà hai model rẻ
    bất đồng.
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

        rm = self.runners[TARGET_MODEL](page_id)
        rp = self.runners[CHEAP_EXTERNAL](page_id)
        s_mp = pair_sims(rm, rp)

        # Subtask nào target ~ Mistral OCR thì đã là Easy -> không cần model đắt.
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
            src[t] = label_source(tier)
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
