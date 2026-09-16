"""§3.3 — "Automated QA tools ... to maintain annotation consistency" (dòng 64).

Paper chỉ có đúng một câu đó, không nói QA cái gì. Đây là chỗ mơ hồ nhất của
§3.3, nên ghi rõ: toàn bộ file này là diễn giải tự đề xuất. Ba nhóm kiểm tra
dưới đây chọn theo tiêu chí "rẻ, tất định, bắt được lỗi mà người hay mắc thật",
KHÔNG dùng model nào:

  A. Tính hợp lệ về dạng   — nhãn có dựng lại được thành ảnh không (LaTeX
     compile được? HTML ra đúng lưới?). Tái dùng render.py, không viết mới.
  B. Tỉnh táo về nội dung  — nhãn rỗng, nhãn y hệt bản nháp đã biết là sai,
     nhãn dài/ngắn bất thường so với vùng ảnh.
  C. Nhất quán giữa người  — trộn lặp một tỉ lệ nhỏ mẫu cho 2 annotator rồi
     đo tương đồng. Đây mới đúng nghĩa đen "annotation consistency"; A và B
     là điều kiện cần, C mới phát hiện được annotator trôi chuẩn theo thời
     gian hoặc hiểu sai hướng dẫn.

Nguyên tắc chung: QA chỉ BÁO, không tự sửa và không tự loại. Nhãn do người
làm ra là thứ đắt nhất trong cả pipeline; máy nghi ngờ thì đưa người xem lại,
không được im lặng vứt đi.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .metrics import SIM_FN, norm_text, parse_table
from .render import render_label

# Ngưỡng cờ "dài bất thường": số ký tự trên 1000 px² vùng ảnh. Rất rộng, chỉ
# nhằm bắt lỗi thô kiểu dán nhầm cả trang vào một ô bảng.
MAX_CHARS_PER_KPX2 = 12.0


@dataclass
class QAIssue:
    code: str            # định danh ngắn để đếm/nhóm
    detail: str

    def __str__(self) -> str:
        return f"{self.code}: {self.detail}"


@dataclass
class QAResult:
    page_id: str
    subtask: str
    issues: List[QAIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues


# ------------------------------------------------------- A. hợp lệ về dạng --

def check_format(content: str, subtask: str, dpi: int = 100) -> List[QAIssue]:
    """Nhãn phải dựng lại được thành ảnh. Không dựng được = gần như chắc chắn
    cấu trúc hỏng (trừ khi backend yếu — xem cảnh báo mathtext trong render.py,
    vì vậy code lỗi ghi kèm backend để phân biệt)."""
    issues: List[QAIssue] = []
    if not (content or "").strip():
        return [QAIssue("rỗng", "nhãn rỗng")]

    r = render_label(content, subtask, dpi)
    if r is not None and not r.ok:
        issues.append(QAIssue("không dựng được",
                              f"backend={r.backend} lỗi={r.error[:120]}"))

    if subtask == "table":
        rows = parse_table(content)
        if not rows:
            issues.append(QAIssue("bảng không parse được", "không tách được hàng/ô nào"))
        else:
            widths = {sum(cs for _rs, cs, _c in r_) for r_ in rows}
            if len(widths) > 1:
                issues.append(QAIssue("lưới lệch",
                                      f"số cột không đồng nhất giữa các hàng: {sorted(widths)}"))
    return issues


# ---------------------------------------------------- B. tỉnh táo nội dung --

def check_content(content: str, subtask: str,
                  box_area_px: Optional[float] = None,
                  known_bad: Optional[str] = None) -> List[QAIssue]:
    """`known_bad` là bản nháp mà tầng tự động đã kết luận là sai — nhãn người
    trùng khít với nó nghĩa là annotator bấm 'chấp nhận' mà không thực sự sửa."""
    issues: List[QAIssue] = []
    s = norm_text(content)
    if not s:
        return [QAIssue("rỗng", "nhãn rỗng sau khi chuẩn hoá")]

    if known_bad:
        sim_fn = SIM_FN.get(subtask, SIM_FN["text"])
        if sim_fn(content, known_bad) >= 0.999:
            issues.append(QAIssue("y hệt bản đã biết sai",
                                  "nhãn trùng khít bản nháp mà tầng tự động đã báo lỗi"))

    if box_area_px and box_area_px > 0:
        density = len(s) / (box_area_px / 1000.0)
        if density > MAX_CHARS_PER_KPX2:
            issues.append(QAIssue("dài bất thường",
                                  f"{len(s)} ký tự trên {box_area_px:,.0f}px² "
                                  f"({density:.1f}/1000px², ngưỡng {MAX_CHARS_PER_KPX2})"))

    if "[?]" in content:
        issues.append(QAIssue("còn dấu [?]",
                              "pre-annotation đánh dấu chỗ không đọc được, người chưa xử lý"))
    return issues


def check_annotation(content: str, subtask: str, page_id: str = "",
                     box_area_px: Optional[float] = None,
                     known_bad: Optional[str] = None) -> QAResult:
    """Chạy cả nhóm A và B cho một nhãn."""
    issues = check_format(content, subtask)
    if not any(i.code == "rỗng" for i in issues):   # rỗng thì nhóm B không thêm được gì
        issues += check_content(content, subtask, box_area_px, known_bad)
    return QAResult(page_id, subtask, issues)


# ------------------------------------------- C. nhất quán giữa annotator ----

def double_annotation_sample(n_total: int, rate: float = 0.05,
                             seed: int = 0) -> List[int]:
    """Chọn ngẫu nhiên các vị trí cần giao cho 2 annotator làm độc lập.

    5% là mức thường dùng để vừa đủ phát hiện annotator trôi chuẩn mà không đội
    chi phí đáng kể — CHƯA hiệu chỉnh cho dự án này, chỉnh sau khi biết số
    annotator thật và độ tản của họ.
    """
    import numpy as np
    k = max(1, int(round(n_total * rate))) if n_total else 0
    if k == 0:
        return []
    rng = np.random.default_rng(seed)
    return sorted(int(i) for i in rng.choice(n_total, min(k, n_total), replace=False))


def inter_annotator_agreement(pairs: Sequence[Tuple[str, str, str]]) -> Dict[str, object]:
    """`pairs` = [(subtask, nhãn_của_A, nhãn_của_B), ...] trên phần mẫu trộn lặp.

    Trả tương đồng trung bình theo subtask + danh sách cặp lệch nhiều nhất.
    Cặp lệch mạnh là thứ cần người thứ ba phân xử, ĐỒNG THỜI là tín hiệu hướng
    dẫn chú thích đang mơ hồ ở chỗ nào — sửa hướng dẫn rẻ hơn sửa từng nhãn.
    """
    import numpy as np
    by_st: Dict[str, List[float]] = {}
    scored: List[Tuple[float, str, str, str]] = []
    for subtask, a, b in pairs:
        sim_fn = SIM_FN.get(subtask, SIM_FN["text"])
        s = float(sim_fn(a, b))
        by_st.setdefault(subtask, []).append(s)
        scored.append((s, subtask, a, b))
    scored.sort(key=lambda x: x[0])
    return {
        "n": len(pairs),
        "theo_subtask": {k: {"n": len(v), "tương đồng tb": float(np.mean(v))}
                         for k, v in sorted(by_st.items())},
        "lệch nhất": [{"subtask": st, "sim": round(s, 3), "A": a[:80], "B": b[:80]}
                      for s, st, a, b in scored[:10]],
    }


def qa_summary(results: Sequence[QAResult]) -> Dict[str, object]:
    """Gộp báo cáo — đếm theo mã lỗi để biết nên sửa hướng dẫn chú thích ở đâu."""
    counts: Dict[str, int] = {}
    for r in results:
        for iss in r.issues:
            counts[iss.code] = counts.get(iss.code, 0) + 1
    n_bad = sum(1 for r in results if not r.ok)
    return {"tổng": len(results), "có vấn đề": n_bad,
            "% sạch": 100.0 * (len(results) - n_bad) / max(1, len(results)),
            "theo mã lỗi": dict(sorted(counts.items(), key=lambda kv: -kv[1]))}
