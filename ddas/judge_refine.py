"""§3.3 — Annotation Pipeline for Hard Case: Judge-and-Refine + hàng đợi người.

Đầu vào là `hard_queue` (List[HardItem]) do sft.py tách ra: các mẫu mà 3 model
CMCV bất đồng, nên KHÔNG có nhãn đồng thuận nào đáng tin. Dùng thẳng chúng để
train sẽ bơm nhiễu vào tập nhãn (paper dòng 45). Module này làm hai tầng:

  tầng 1 — Judge-and-Refine tự động (render-then-verify, xem render.py):
      mỗi vòng: render nhãn hiện tại -> đặt cạnh ảnh gốc -> model trọng tài
      khoanh lỗi và đề xuất bản sửa -> render lại. Thoát khi trọng tài không
      còn thấy lỗi (=> có nhãn dùng được), hoặc khi vòng lặp bí.
  tầng 2 — mẫu tầng 1 không cứu được thì xếp hàng chú thích tay, ưu tiên theo
      đúng 2 tiêu chí paper (dòng 61-62): correction efficiency rồi marginal
      impact. Xem `prioritize()`.

Chọn model trọng tài — CHỦ Ý LỆCH so với paper, đọc kỹ trước khi đổi:
paper dùng Qwen3-VL-235B và lập luận rằng nó "độc lập với CMCV model pool".
Lập luận đó vốn đã yếu (pool của paper có Qwen3-VL-30B — cùng dòng model, lỗi
tương quan là chuyện bình thường), và ở repo này thì sai hẳn: TARGET_MODEL
chính là Qwen3-VL-8B, còn Gemini 3 Pro đã làm trọng tài trong CMCV. Nếu trọng
tài §3.3 cũng thuộc hai dòng đó, nó sẽ "đồng cảm" đúng những lỗi mà CMCV đã bỏ
sót — tức tầng sửa lỗi mù đúng chỗ cần sáng nhất. Nên mặc định lấy dòng thứ tư
(OpenAI), đúng gợi ý đã ghi ở cmcv.py:20.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image

from .cmcv import CMCVRecord, Tier
from .config import JudgeRefineConfig
from .element import crop_box
from .metrics import SIM_FN
from .render import Render, render_label
from .sft import HardItem, SFTRecord

JUDGE_MODEL = "gpt-5"          # khác dòng với cả 3 model CMCV — xem docstring đầu file

# Lý do một mẫu rơi xuống tầng người.
REASON_NO_DRAFT = "no_draft"            # không model nào có nội dung -> không có gì để sửa
REASON_NO_JUDGE_PATH = "no_judge_path"  # subtask không chạy được vòng judge (layout)
REASON_RENDER_FAILED = "render_failed"  # không dựng được ảnh -> không xác minh được
REASON_NO_FIX = "no_fix_proposed"       # trọng tài thấy lỗi nhưng không đề xuất được bản sửa
REASON_STUCK = "stuck"                  # sửa mà gần như không đổi, vẫn báo lỗi
REASON_OSCILLATING = "oscillating"      # quay vòng giữa các phương án cũ
REASON_MAX_ROUNDS = "max_rounds"        # hết ngân sách vòng lặp

# Tiêu chí ưu tiên #1 của paper: trọng tài ĐÃ khoanh được lỗi nhưng khâu sửa bó
# tay. Người chú thích chỉ phải sửa cục bộ tại chỗ đã khoanh -> năng suất cao
# nhất. REASON_RENDER_FAILED/NO_DRAFT không thuộc nhóm này: ở đó chưa ai biết
# lỗi nằm đâu, người phải làm lại từ đầu.
LOCATED_BUT_UNFIXED = (REASON_NO_FIX, REASON_STUCK, REASON_OSCILLATING, REASON_MAX_ROUNDS)


@dataclass
class JudgeVerdict:
    """Đầu ra một vòng của model trọng tài."""
    has_error: bool
    confidence: float = 0.0             # [0,1] — độ chắc chắn của phán đoán "có lỗi"
    corrected: Optional[str] = None     # None = thấy lỗi nhưng không sửa nổi
    note: str = ""                      # mô tả lỗi, chuyển cho người chú thích


@dataclass
class RefinedRecord:
    """Mẫu Hard đã được sửa tới mức trọng tài không còn thấy lỗi."""
    subtask: str
    page_id: str
    content: str
    rounds: int
    confidence: float
    judge_model: str
    box: Optional[np.ndarray] = None
    element_id: Optional[int] = None

    def to_sft(self) -> SFTRecord:
        """Nhập vào SFT set, GIỮ NGUYÊN tier HARD để truy vết được: đây là nhãn
        do trọng tài sửa, không phải nhãn đồng thuận model như Easy/Medium."""
        return SFTRecord(self.subtask, self.page_id, Tier.HARD,
                         f"judge-refine:{self.judge_model}", self.content,
                         self.box, self.element_id)


@dataclass
class ExpertItem:
    """Mẫu tầng 1 không cứu được -> chuyển người, kèm đủ dấu vết để xếp ưu tiên."""
    subtask: str
    page_id: str
    reason: str
    confidence: float = 0.0
    rounds: int = 0
    draft: Optional[str] = None         # bản tốt nhất tầng 1 đạt được, để người sửa tiếp
    note: str = ""
    backend: str = ""                   # backend render đã dùng (chạy lại được khi có TeX thật)
    box: Optional[np.ndarray] = None
    element_id: Optional[int] = None

    @property
    def located(self) -> bool:
        return self.reason in LOCATED_BUT_UNFIXED


Outcome = Union[RefinedRecord, ExpertItem]

# judge_fn(ảnh gốc, ảnh render | None, nội dung hiện tại, subtask) -> JudgeVerdict
JudgeFn = Callable[[Image.Image, Optional[Image.Image], str, str], JudgeVerdict]


class JudgeRefine:
    """Vòng lặp render-then-verify trên hàng đợi Hard.

    `judge_fn` và `image_fn` được inject giống mọi lời gọi nặng khác trong repo
    (xem DDASPipeline): test chạy hàm giả, production cắm API model trọng tài và
    hàm nạp ảnh trang, dùng chung đúng một đường code.
    """

    def __init__(self, judge_fn: JudgeFn,
                 image_fn: Callable[[str], Image.Image],
                 cfg: JudgeRefineConfig | None = None,
                 judge_model: str = JUDGE_MODEL):
        self.judge_fn = judge_fn
        self.image_fn = image_fn
        self.cfg = cfg or JudgeRefineConfig()
        self.judge_model = judge_model
        self.stats: Dict[str, int] = {"items": 0, "judge_calls": 0, "rounds": 0,
                                      "resolved": 0, "expert": 0, "render_failed": 0}

    # ------------------------------------------------------------ 1 mẫu ----
    def run_item(self, item: HardItem) -> Outcome:
        self.stats["items"] += 1
        content = item.draft

        if not content:
            return self._to_expert(item, REASON_NO_DRAFT)
        if item.subtask not in self.cfg.judgeable:
            return self._to_expert(item, REASON_NO_JUDGE_PATH, draft=content)

        page_img = self.image_fn(item.page_id)
        orig = crop_box(page_img, item.box) if item.box is not None else page_img
        renderable = item.subtask in self.cfg.renderable
        sim_fn = SIM_FN.get(item.subtask, SIM_FN["text"])
        seen = {content}
        backend = ""

        for rnd in range(1, max(1, self.cfg.max_rounds) + 1):
            shot: Optional[Image.Image] = None
            if renderable:
                r: Render = render_label(content, item.subtask, self.cfg.dpi,
                                         self.cfg.latex_backends)
                backend = r.backend
                if not r.ok:
                    # Không dựng được ảnh thì KHÔNG xác minh được, mà cũng không
                    # kết luận được là nhãn sai (có thể chỉ do backend yếu — xem
                    # cảnh báo mathtext trong render.py). Đẩy người, ghi backend.
                    self.stats["render_failed"] += 1
                    return self._to_expert(item, REASON_RENDER_FAILED, rounds=rnd - 1,
                                           draft=content, note=r.error, backend=backend)
                shot = r.image

            self.stats["judge_calls"] += 1
            self.stats["rounds"] += 1
            v = self.judge_fn(orig, shot, content, item.subtask)

            if not v.has_error:
                self.stats["resolved"] += 1
                return RefinedRecord(item.subtask, item.page_id, content, rnd,
                                     v.confidence, self.judge_model,
                                     item.box, item.element_id)
            if v.corrected is None:
                return self._to_expert(item, REASON_NO_FIX, v.confidence, rnd,
                                       content, v.note, backend)
            if sim_fn(content, v.corrected) >= self.cfg.converge_tau:
                # Vẫn báo lỗi nhưng bản "sửa" không khác gì bản cũ -> bí thật,
                # chạy thêm vòng chỉ tốn tiền.
                return self._to_expert(item, REASON_STUCK, v.confidence, rnd,
                                       v.corrected, v.note, backend)
            if v.corrected in seen:
                return self._to_expert(item, REASON_OSCILLATING, v.confidence, rnd,
                                       v.corrected, v.note, backend)
            seen.add(v.corrected)
            content = v.corrected

        return self._to_expert(item, REASON_MAX_ROUNDS, v.confidence, rnd,
                               content, v.note, backend)

    def _to_expert(self, item: HardItem, reason: str, confidence: float = 0.0,
                   rounds: int = 0, draft: Optional[str] = None, note: str = "",
                   backend: str = "") -> ExpertItem:
        self.stats["expert"] += 1
        return ExpertItem(item.subtask, item.page_id, reason, confidence, rounds,
                          draft, note, backend, item.box, item.element_id)

    # ----------------------------------------------------- cả hàng đợi ----
    def run(self, queue: Sequence[HardItem]) -> Tuple[List[RefinedRecord], List[ExpertItem]]:
        refined: List[RefinedRecord] = []
        expert: List[ExpertItem] = []
        for item in queue:
            out = self.run_item(item)
            (refined if isinstance(out, RefinedRecord) else expert).append(out)
        return refined, expert

    @property
    def resolve_rate(self) -> float:
        """Tỉ lệ mẫu Hard được tự động cứu — con số quyết định ngân sách người."""
        return self.stats["resolved"] / max(1, self.stats["items"])

    @property
    def mean_rounds(self) -> float:
        return self.stats["rounds"] / max(1, self.stats["items"])


# ------------------------------------------------- xếp ưu tiên cho người ----

def weakness_by_subtask(records: Sequence[CMCVRecord]) -> Dict[str, float]:
    """Độ yếu của target model theo subtask = 1 - đồng thuận trung bình với external.

    Tiêu chí ưu tiên #2 của paper (marginal impact): dồn ngân sách chú thích vào
    subtask model đang yếu nhất. Không cần đo gì thêm — tái dùng đúng `sims` mà
    CMCV (§3.2) đã tính trên candidate set.
    """
    acc: Dict[str, List[float]] = {}
    for r in records:
        for st, s in r.sims.items():
            vals = [v for k, v in s.items() if k in ("M-P", "M-Q") and v is not None]
            if vals:
                acc.setdefault(st, []).append(float(np.mean(vals)))
    return {st: 1.0 - float(np.mean(v)) for st, v in acc.items()}


def prioritize(items: Sequence[ExpertItem],
               weakness: Optional[Dict[str, float]] = None,
               min_confidence: float = 0.70,
               budget: Optional[int] = None) -> List[ExpertItem]:
    """Xếp hàng đợi chú thích tay theo đúng 2 tiêu chí §3.3 (dòng 61-62).

    1. correction efficiency — trọng tài đã khoanh lỗi CHẮC CHẮN nhưng không sửa
       được: người chỉ cần sửa cục bộ tại chỗ đã khoanh, năng suất cao nhất.
    2. marginal impact — trong nhóm đó, ưu tiên subtask model đang yếu nhất
       (theo `weakness`), để mỗi mẫu chú thích đóng góp biên nhiều nhất.

    `budget` cắt còn N mẫu đầu (JudgeRefineConfig.expert_budget, paper: 192K).
    """
    w = weakness or {}
    ranked = sorted(
        items,
        key=lambda x: (0 if (x.located and x.confidence >= min_confidence) else 1,
                       -w.get(x.subtask, 0.0), -x.confidence),
    )
    return ranked[:budget] if budget else ranked


def summarize(refined: Sequence[RefinedRecord], expert: Sequence[ExpertItem]) -> Dict[str, Dict[str, object]]:
    """Thống kê theo subtask — dùng để báo cáo và để dò ngưỡng cấu hình."""
    out: Dict[str, Dict[str, object]] = {}
    subtasks = {r.subtask for r in refined} | {e.subtask for e in expert}
    for st in sorted(subtasks):
        rs = [r for r in refined if r.subtask == st]
        es = [e for e in expert if e.subtask == st]
        reasons: Dict[str, int] = {}
        for e in es:
            reasons[e.reason] = reasons.get(e.reason, 0) + 1
        out[st] = {"refined": len(rs), "expert": len(es),
                   "resolve_rate": len(rs) / max(1, len(rs) + len(es)),
                   "vòng tb": float(np.mean([r.rounds for r in rs])) if rs else 0.0,
                   "lý do": reasons}
    return out
