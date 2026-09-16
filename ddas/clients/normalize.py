"""Chuẩn hoá output 3 nhà cung cấp về đúng một `ParseResult`.

Đây là chỗ dễ sai âm thầm nhất trong cả lớp client, vì hai lý do dưới đây đều
KHÔNG gây exception — chúng chỉ làm điểm đồng thuận tụt xuống và đẩy dữ liệu
sạch xuống tier Hard, tức là tiêu tiền §3.3 cho những trang vốn không có vấn đề.

  1. NHÃN PHẢI KHỚP CHUỖI CHÍNH XÁC. metrics.layout_sim ghép hai bbox chỉ khi
     `la == lb` (xem metrics.py:200). Mistral OCR trả 13 loại block của nó,
     PaddleOCR-VL trả bộ nhãn của PaddleX, Qwen3-VL trả đúng thứ ta bảo nó
     trả. Không quy về một bộ thì layout_sim ~ 0 trên MỌI trang, và taxonomy
     Easy/Medium/Hard của §3.2 sụp hoàn toàn ở subtask layout.
  2. TOẠ ĐỘ PHẢI CÙNG HỆ QUY CHIẾU. element.crop_box cắt ảnh bằng toạ độ PIXEL
     TUYỆT ĐỐI trên ảnh do PageStore trả về (đã resize, xem pagestore.py). VLM
     thường trả toạ độ chuẩn hoá 0-1000 (quy ước Qwen) hoặc 0-1. Quên đổi thì
     IoU vẫn tính ra số (không crash) nhưng là số vô nghĩa.

Bộ nhãn chuẩn lấy ĐÚNG theo layout_heron.py::_TO_PARSE_LABEL, không tự chế bộ
mới: Heron là nguồn bbox của Stage 2 và element.py ghép nội dung CMCV vào box
Heron theo IoU + nhãn, nên hai bên lệch taxonomy là hỏng khâu ghép.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..cmcv import ParseResult

log = logging.getLogger("ddas.clients.normalize")

# Bộ nhãn chuẩn của hệ thống — khớp layout_heron.py.
CANON_LABELS = ("text", "title", "list", "caption", "formula", "table")

# LƯU Ý đã biết, cố ý giữ: Heron map "Picture" -> "text" (Picture không nằm
# trong _TO_PARSE_LABEL nên rơi vào default). Ở đây map figure/picture/image
# -> "text" cho KHỚP Heron thay vì tự thêm nhãn "figure" — lệch taxonomy giữa
# hai module còn tệ hơn là cùng sai một kiểu. Muốn sửa thì sửa CẢ HAI chỗ.
_ALIASES: Dict[str, str] = {
    # dạng chung
    "text": "text", "paragraph": "text", "plain text": "text", "body": "text",
    "title": "title", "doc_title": "title", "heading": "title",
    "section-header": "title", "section_header": "title", "sub_title": "title",
    "list": "list", "list-item": "list", "list_item": "list", "listitem": "list",
    "caption": "caption", "figure_caption": "caption", "table_caption": "caption",
    "figcaption": "caption", "image_caption": "caption",
    "formula": "formula", "equation": "formula", "isolate_formula": "formula",
    "interline_equation": "formula", "display_formula": "formula", "math": "formula",
    "table": "table", "table_body": "table", "tablebody": "table",
    # figure/picture -> text, xem ghi chú ở trên
    "figure": "text", "picture": "text", "image": "text", "chart": "text",
    # header/footer/footnote/code/form: Heron gộp hết vào "text"
    "page-header": "text", "page_header": "text", "header": "text",
    "page-footer": "text", "page_footer": "text", "footer": "text",
    "footnote": "text", "code": "text", "algorithm": "text", "form": "text",
    "key-value region": "text", "key_value": "text", "document index": "text",
    "abstract": "text", "reference": "text", "seal": "text", "stamp": "text",
    "checkbox-selected": "text", "checkbox-unselected": "text",
}


def canon_label(raw: str) -> str:
    """Quy nhãn bất kỳ của nhà cung cấp về bộ nhãn chuẩn. Không biết -> 'text'."""
    if not raw:
        return "text"
    k = re.sub(r"[\s_\-]+", " ", str(raw).strip().lower())
    if k in _ALIASES:
        return _ALIASES[k]
    k2 = k.replace(" ", "-")
    if k2 in _ALIASES:
        return _ALIASES[k2]
    k3 = k.replace(" ", "_")
    if k3 in _ALIASES:
        return _ALIASES[k3]
    log.debug("nhãn lạ %r -> 'text'", raw)
    return "text"


def to_abs_box(box: Sequence[float], width: int, height: int,
               space: str = "auto") -> Optional[np.ndarray]:
    """Đổi bbox về pixel tuyệt đối xyxy trên ảnh (width, height).

    `space`: "abs" | "norm1" (0-1) | "norm1000" (0-1000) | "auto".
    "auto" đoán theo biên độ giá trị — tiện cho pilot, nhưng khi chạy thật nên
    CHỈ ĐỊNH RÕ: một trang toàn bbox nhỏ nằm ở góc trên-trái có thể bị đoán
    nhầm là toạ độ chuẩn hoá, và lỗi đó không có cách nào phát hiện tự động.
    """
    if box is None or len(box) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in box)
    except (TypeError, ValueError):
        return None
    if not all(np.isfinite(v) for v in (x1, y1, x2, y2)):
        return None

    if space == "auto":
        m = max(abs(x1), abs(y1), abs(x2), abs(y2))
        space = "norm1" if m <= 1.5 else ("norm1000" if m <= 1000.5 else "abs")
    if space == "norm1":
        x1, x2 = x1 * width, x2 * width
        y1, y2 = y1 * height, y2 * height
    elif space == "norm1000":
        x1, x2 = x1 / 1000.0 * width, x2 / 1000.0 * width
        y1, y2 = y1 / 1000.0 * height, y2 / 1000.0 * height

    # Một số model trả toạ độ đảo; chuẩn hoá về x1<x2, y1<y2 rồi chặn biên.
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1, y1 = max(0.0, x1), max(0.0, y1)
    x2, y2 = min(float(width), x2), min(float(height), y2)
    if x2 - x1 < 1.0 or y2 - y1 < 1.0:      # box rỗng/suy biến -> bỏ
        return None
    return np.array([x1, y1, x2, y2], dtype=np.float32)


_HTML_TABLE = re.compile(r"<table\b.*?</table>", re.S | re.I)
_LATEX_INLINE = re.compile(r"\$\$(.+?)\$\$|\\\[(.+?)\\\]", re.S)


def extract_tables_html(markdown: str) -> List[str]:
    """Bóc bảng HTML khỏi markdown. Bảng dạng pipe-markdown được đổi sang HTML
    để cùng một hệ quy chiếu với TEDS (metrics.table_sim nhận HTML)."""
    out = [m.group(0) for m in _HTML_TABLE.finditer(markdown or "")]
    out.extend(_pipe_tables_to_html(markdown or ""))
    return out


def _pipe_tables_to_html(md: str) -> List[str]:
    tables, block = [], []
    for line in md.splitlines():
        if line.strip().startswith("|") and line.strip().endswith("|"):
            block.append(line.strip())
            continue
        if block:
            html = _pipe_block_to_html(block)
            if html:
                tables.append(html)
            block = []
    if block:
        html = _pipe_block_to_html(block)
        if html:
            tables.append(html)
    return tables


def _pipe_block_to_html(lines: List[str]) -> Optional[str]:
    rows = []
    for ln in lines:
        if re.fullmatch(r"\|[\s:\-\|]+\|", ln):   # dòng phân cách ---|---
            continue
        rows.append([c.strip() for c in ln.strip("|").split("|")])
    if len(rows) < 2:                              # 1 dòng thì không phải bảng
        return None
    head = "".join(f"<td>{c}</td>" for c in rows[0])
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows[1:])
    return f"<table><tr>{head}</tr>{body}</table>"


def extract_formulas_latex(markdown: str) -> List[str]:
    """Bóc công thức display khỏi markdown ($$..$$ hoặc \\[..\\])."""
    out = []
    for m in _LATEX_INLINE.finditer(markdown or ""):
        s = (m.group(1) or m.group(2) or "").strip()
        if s:
            out.append(s)
    return out


def blocks_to_parse_result(page_id: str, model: str, blocks: Iterable[Dict[str, Any]],
                           width: int, height: int, *, box_space: str = "auto",
                           latency_ms: float = 0.0) -> ParseResult:
    """Gộp danh sách block đã chuẩn hoá thành ParseResult.

    Thứ tự block đầu vào ĐƯỢC COI LÀ thứ tự đọc — cmcv._seq_sim so formula/table
    theo thứ tự nên nhà cung cấp trả lộn xộn sẽ bị phạt oan. Mọi adapter phải
    sắp theo thứ tự đọc trước khi gọi vào đây.
    """
    boxes: List[np.ndarray] = []
    labels: List[str] = []
    contents: List[str] = []
    formulas: List[str] = []
    tables: List[str] = []
    text_parts: List[str] = []

    for b in blocks:
        lab = canon_label(b.get("type") or b.get("label") or b.get("category") or "")
        content = (b.get("content") or b.get("text") or "").strip()
        box = to_abs_box(b.get("bbox") or b.get("box") or (), width, height, box_space)
        if box is not None:
            boxes.append(box)
            labels.append(lab)
            # Căn 1-1 với boxes/labels — Stage 2 so nội dung theo từng vùng,
            # chuỗi text ghép ở dưới KHÔNG tách lại được. Xem ParseResult.contents.
            contents.append(content)
        if not content:
            continue
        if lab == "formula":
            formulas.append(content)
        elif lab == "table":
            tables.append(content if "<" in content else
                          (_pipe_block_to_html(content.splitlines()) or content))
        else:
            text_parts.append(content)

    return ParseResult(
        page_id=page_id, model=model,
        text="\n".join(text_parts),
        boxes=(np.stack(boxes) if boxes else np.zeros((0, 4), np.float32)),
        labels=labels, contents=contents, formulas=formulas, tables=tables,
        latency_ms=latency_ms)


def is_empty_result(pr: ParseResult) -> bool:
    """ParseResult không mang một mẩu nội dung nào.

    Đây KHÔNG phải kiểm tra vặt. cmcv.pair_sims coi hai kết quả rỗng là giống
    nhau tuyệt đối (mọi sim = 1.0), nên nếu cả ba model cùng trả rỗng thì trang
    được gán EASY và "nhãn đồng thuận" rỗng đi thẳng vào tập train — mà cascade
    còn cắt luôn model thứ ba nên không có ai phản biện. Đo được điều này bằng
    testkit/demo_clients.py bài [6b].
    """
    return (not pr.text.strip() and not pr.formulas and not pr.tables
            and len(pr.boxes) == 0)


def markdown_to_parse_result(page_id: str, model: str, markdown: str, *,
                             latency_ms: float = 0.0) -> ParseResult:
    """Đường dự phòng khi nhà cung cấp chỉ trả markdown phẳng, KHÔNG có bbox.

    Kết quả không có boxes => layout_sim với model này luôn = 0. Nên chỉ dùng
    khi model đó không có đường ra bbox; nếu có thì luôn ưu tiên đường block.
    """
    tables = extract_tables_html(markdown)
    formulas = extract_formulas_latex(markdown)
    text = _HTML_TABLE.sub(" ", markdown or "")
    text = _LATEX_INLINE.sub(" ", text)
    return ParseResult(page_id=page_id, model=model, text=text.strip(),
                       formulas=formulas, tables=tables, latency_ms=latency_ms)
