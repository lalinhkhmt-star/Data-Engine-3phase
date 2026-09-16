"""§3.3 tầng người — AI pre-annotation + gói việc cho chuyên gia soát (dòng 64).

Paper: "Human annotation follows an AI pre-annotation and expert review-and-
correction workflow. Gemini 3 Pro is used for pre-annotation because of its
strong multimodal reasoning capability and its independence from the CMCV
model pool, thereby avoiding data leakage."

Chỉ chừng đó, nên mấy quyết định dưới đây là tự đề xuất:

1. PRE-ANNOTATE TỪ ẢNH GỐC, KHÔNG CHO XEM BẢN NHÁP HỎNG. Cách làm hiển nhiên
   hơn là đưa `ExpertItem.draft` cho model "sửa tiếp", nhưng làm vậy thì bản
   pre-annotation kế thừa đúng thiên lệch của bản nháp — mà bản nháp đó vừa
   thất bại qua nhiều vòng Judge-and-Refine rồi. Neo vào ảnh gốc thì được một
   ý kiến ĐỘC LẬP thật sự, đúng tinh thần "avoiding data leakage" của dòng 64.

2. NGƯỜI NHẬN ĐƯỢC CẢ HAI PHƯƠNG ÁN + CHỖ NGHI LỖI. Mỗi ExpertTask gồm: bản
   nháp tốt nhất của Judge-and-Refine, bản pre-annotation độc lập của Gemini,
   ghi chú khoanh vùng lỗi của trọng tài, và điểm tương đồng giữa hai bản.
   Hai bản ĐỒNG THUẬN => nhiều khả năng đúng, người chỉ soát nhanh. Hai bản
   LỆCH => người phải phân xử, nhưng biết ngay lệch ở đâu mà nhìn. Cả hai
   trường hợp đều nhanh hơn gõ lại từ đầu, đúng mục tiêu "maximizing
   annotation throughput" (dòng 61).

3. Model: PREANNOT_MODEL = Gemini 3 Pro đúng như paper — hợp lệ ở repo này vì
   pool CMCV là Qwen3-VL-8B / Mistral OCR / PaddleOCR-VL và trọng tài §3.3 là
   GPT-5, nên Gemini không đụng vai nào khác (xem cmcv.py, judge_refine.py).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import numpy as np
from PIL import Image

from .element import crop_box
from .judge_refine import CallModel, ExpertItem
from .metrics import SIM_FN

PREANNOT_MODEL = "gemini-3-pro"      # dòng 64 paper; rảnh vai vì không nằm trong pool CMCV

# preannot_fn(ảnh gốc, subtask) -> nội dung chú thích đề xuất ("" nếu model bó tay)
PreannotFn = Callable[[Image.Image, str], str]


@dataclass
class ExpertTask:
    """Một gói việc hoàn chỉnh đưa cho chuyên gia người."""
    subtask: str
    page_id: str
    reason: str                       # vì sao tầng tự động không cứu được
    note: str                         # trọng tài khoanh lỗi ở đâu
    draft: Optional[str]              # phương án A — bản tốt nhất của Judge-and-Refine
    preannot: Optional[str]           # phương án B — pre-annotation độc lập
    agreement: Optional[float]        # tương đồng A/B; None khi thiếu một trong hai
    confidence: float = 0.0           # độ tin cậy phán đoán "có lỗi" của trọng tài
    box: Optional[np.ndarray] = None
    element_id: Optional[int] = None

    @property
    def mode(self) -> str:
        """Kiểu việc — dùng để phân luồng cho annotator và ước công.

        'soát nhanh'  : hai phương án độc lập trùng khớp -> gần như chắc đúng.
        'phân xử'     : hai phương án lệch nhau -> người chọn/ghép.
        'gõ lại'      : chỉ có nhiều nhất một phương án -> nặng nhất.
        """
        if self.agreement is None:
            return "gõ lại"
        return "soát nhanh" if self.agreement >= 0.98 else "phân xử"


def make_preannot_fn(call_model: CallModel) -> PreannotFn:
    """Dựng PreannotFn từ hàm gọi model, dùng prompt trong prompts.py."""
    from .prompts import PREANNOT_SYSTEM, preannot_prompt

    def preannot_fn(image: Image.Image, subtask: str) -> str:
        return (call_model(PREANNOT_SYSTEM, preannot_prompt(subtask), [image]) or "").strip()

    return preannot_fn


def build_expert_task(item: ExpertItem, image_fn: Callable[[str], Image.Image],
                      preannot_fn: Optional[PreannotFn] = None) -> ExpertTask:
    """Gói một ExpertItem thành việc cho người, kèm pre-annotation độc lập.

    `preannot_fn=None` -> bỏ qua bước AI pre-annotation (chạy được khi chưa nối
    model, hoặc khi muốn đo xem pre-annotation có thực sự tăng năng suất không).
    """
    preannot = None
    if preannot_fn is not None:
        page = image_fn(item.page_id)
        crop = crop_box(page, item.box) if item.box is not None else page
        preannot = preannot_fn(crop, item.subtask) or None

    agreement = None
    if preannot and item.draft:
        sim_fn = SIM_FN.get(item.subtask, SIM_FN["text"])
        agreement = float(sim_fn(item.draft, preannot))

    return ExpertTask(item.subtask, item.page_id, item.reason, item.note,
                      item.draft, preannot, agreement, item.confidence,
                      item.box, item.element_id)


def build_expert_batch(items: Sequence[ExpertItem],
                       image_fn: Callable[[str], Image.Image],
                       preannot_fn: Optional[PreannotFn] = None) -> List[ExpertTask]:
    """Chạy trên hàng đợi ĐÃ xếp ưu tiên (judge_refine.prioritize) — thứ tự giữ
    nguyên, vì pre-annotation không được phép đảo lại ưu tiên đã tính."""
    return [build_expert_task(it, image_fn, preannot_fn) for it in items]


def workload_summary(tasks: Sequence[ExpertTask]) -> dict:
    """Ước khối lượng việc theo kiểu — số liệu để thương lượng ngân sách người."""
    out: dict = {"tổng": len(tasks)}
    for m in ("soát nhanh", "phân xử", "gõ lại"):
        n = sum(1 for t in tasks if t.mode == m)
        out[m] = {"n": n, "%": 100.0 * n / max(1, len(tasks))}
    return out
