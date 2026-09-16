"""Layout detection cho trang SCAN (không có text layer) — Docling Layout Heron.

Lấp chỗ trống 'yolo' để lại trong EmbedConfig.layout_source_order — trước đây
chỉ là placeholder tên gọi (xem costmodel.py RATE['doclayout_yolo']), chưa có
code thật nào chạy được.

Model: docling-project/docling-layout-heron (RT-DETRv2, huấn luyện từ đầu trên
nhiều bộ dữ liệu layout tài liệu) — model layout mặc định của dự án Docling,
detect 17 lớp: Caption, Footnote, Formula, List-item, Page-footer, Page-header,
Picture, Section-header, Table, Text, Title, Document Index, Code,
Checkbox-Selected, Checkbox-Unselected, Form, Key-Value Region.

Output dùng cho 2 việc:
  1. layout_prior_from_heron() — suy ra vector layout-prior 24 chiều GIỐNG
     KHUÔN DẠNG layout_prior.extract_layout_prior() (PDF born-digital) để
     trang scan và trang PDF dùng chung không gian cluster ở Stage 1.
     GIỚI HẠN THẬT (không che giấu, cùng tinh thần layout_prior.py): Heron
     chỉ detect bbox+class, KHÔNG có cỡ chữ — 6 chiều font_stats [18:24]
     luôn bằng 0 cho trang scan, khác trang PDF born-digital có cỡ chữ thật.
  2. Bbox+class trả về (LayoutBox) là nguồn neo bbox ĐỘC LẬP cho CMCV mức
     element (Stage 2), đóng vai trò tương tự text layer PDF ở trang
     born-digital. ĐÃ nối: element.py::derive_element_cmcv() nhận thẳng
     `layout_boxes` từ đây rồi mới tra nội dung của 3 model CMCV theo IoU —
     Heron chạy TRƯỚC, CMCV chạy SAU (xem đầu file element.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
from PIL import Image

from .layout_prior import LAYOUT_DIM, _estimate_columns

CLASSES = [
    "Caption", "Footnote", "Formula", "List-item", "Page-footer", "Page-header",
    "Picture", "Section-header", "Table", "Text", "Title", "Document Index",
    "Code", "Checkbox-Selected", "Checkbox-Unselected", "Form", "Key-Value Region",
]

# Heron -> nhãn dùng chung hệ thống (ParseResult.labels, xem element.py::_elements_of
# chỉ phân biệt 'formula' | 'table' | 'text'/'title'/'list'/'caption').
_TO_PARSE_LABEL = {
    "Formula": "formula", "Table": "table", "Title": "title",
    "List-item": "list", "Caption": "caption",
}
_TEXT_LIKE = {"Text", "Title", "Section-header", "List-item", "Caption", "Footnote",
             "Page-footer", "Page-header", "Code", "Form", "Key-Value Region",
             "Document Index", "Checkbox-Selected", "Checkbox-Unselected"}


@dataclass
class LayoutBox:
    box: np.ndarray          # xyxy, pixel gốc ảnh đầu vào
    cls: str                 # 1 trong 17 lớp Heron
    score: float
    label: str = field(init=False)   # nhãn rút gọn dùng chung hệ thống

    def __post_init__(self):
        self.label = _TO_PARSE_LABEL.get(self.cls, "text")


# ------------------------------------- rule: loại bbox "figure" wrapper thừa -

def _box_area(b: LayoutBox) -> float:
    x1, y1, x2, y2 = b.box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _intersection_area(a: LayoutBox, b: LayoutBox) -> float:
    ax1, ay1, ax2, ay2 = a.box
    bx1, by1, bx2, by2 = b.box
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def drop_redundant_figure_wrappers(
    dets: List[LayoutBox],
    parent_cls: str = "Picture",
    containment_thr: float = 0.85,
    min_children: int = 2,
    area_ratio_thr: float = 0.5,
) -> List[LayoutBox]:
    """Loại bbox "figure" là wrapper thừa đè lên các box con (rule do người dùng chỉ định).

    Docling Layout Heron không có lớp tên đúng "Figure" — lớp gần nghĩa nhất
    trong 17 lớp là "Picture" (xem CLASSES đầu file); đổi `parent_cls` nếu
    bản model bạn dùng đặt tên khác.

    Với mỗi box `parent_cls` (coi là parent):
      1. Tìm mọi box KHÁC trên cùng trang thoả: diện tích nhỏ hơn parent VÀ
         phần giao với parent >= containment_thr * diện tích chính nó (nằm
         gần như trọn vẹn bên trong parent) -> "children".
      2. Số children < min_children -> bỏ qua, GIỮ parent (1 box con lẻ
         thường là chú thích nhỏ trong ảnh thật, không phải wrapper).
      3. (tổng diện tích children) / diện tích parent >= area_ratio_thr
         -> parent là wrapper thừa (chỉ là khung bao quanh 1 cụm ảnh/chữ
         ghép, tự nó không mang thêm thông tin) -> XOÁ parent, giữ children.
      4. Ngược lại -> GIỮ parent (ảnh thật, box nhỏ bên trong là nhãn phụ).

    Đánh giá 1 lượt trên detection GỐC, không đệ quy lại sau khi xoá — 2 box
    Picture lồng nhau (nếu có) được xét độc lập, không theo thứ tự xoá trước/sau.
    """
    parents = [d for d in dets if d.cls == parent_cls]
    drop_ids = set()
    for p in parents:
        pa = _box_area(p)
        if pa <= 0:
            continue
        children = [d for d in dets if d is not p and 0 < _box_area(d) < pa
                   and _intersection_area(p, d) / _box_area(d) >= containment_thr]
        if len(children) < min_children:
            continue
        if sum(_box_area(c) for c in children) / pa >= area_ratio_thr:
            drop_ids.add(id(p))
    return [d for d in dets if id(d) not in drop_ids]


@dataclass
class HeronLayoutDetector:
    """Bọc docling-layout-heron (RT-DETRv2) qua HF transformers.

    Cần: pip install transformers torch pillow (RTDetrV2ForObjectDetection có
    từ transformers>=4.46 — kiểm tra version nếu import lỗi).

    CHƯA ĐO throughput thật trên cụm của bạn. README công bố Heron-101 (bản
    rút gọn) ~28ms/trang trên 1 A100 (~36 trang/giây); bản Heron đầy đủ (model
    này) không có số công bố, ước thận trọng chậm hơn ~2x trong costmodel.py.
    Đổi model_name sang 'docling-project/docling-layout-heron-101' nếu cần ưu
    tiên tốc độ hơn độ chính xác.
    """
    model_name: str = "docling-project/docling-layout-heron"
    device: str = "cpu"
    conf_threshold: float = 0.6
    batch_size: int = 16
    drop_figure_wrappers: bool = True    # xem drop_redundant_figure_wrappers()
    wrapper_parent_cls: str = "Picture"
    wrapper_containment_thr: float = 0.85
    wrapper_min_children: int = 2
    wrapper_area_ratio_thr: float = 0.5
    _processor: object = None
    _model: object = None

    def _lazy_load(self):
        if self._model is not None:
            return
        import torch
        from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection
        self._processor = RTDetrImageProcessor.from_pretrained(self.model_name)
        self._model = RTDetrV2ForObjectDetection.from_pretrained(self.model_name).eval().to(self.device)
        self._torch = torch

    def detect(self, images: List[Image.Image]) -> List[List[LayoutBox]]:
        """Trả về, với mỗi ảnh đầu vào, danh sách LayoutBox đã lọc theo
        conf_threshold rồi lọc tiếp wrapper thừa (nếu drop_figure_wrappers)."""
        self._lazy_load()
        out: List[List[LayoutBox]] = []
        for s in range(0, len(images), self.batch_size):
            batch = images[s:s + self.batch_size]
            inputs = self._processor(images=batch, return_tensors="pt").to(self.device)
            with self._torch.no_grad():
                outputs = self._model(**inputs)
            sizes = self._torch.tensor([im.size[::-1] for im in batch])   # (H, W) mỗi ảnh
            results = self._processor.post_process_object_detection(
                outputs, target_sizes=sizes, threshold=self.conf_threshold)
            id2label = self._model.config.id2label
            for r in results:
                boxes = r["boxes"].cpu().numpy()
                labels = r["labels"].cpu().numpy()
                scores = r["scores"].cpu().numpy()
                page_dets = [LayoutBox(box=boxes[i], cls=id2label[int(labels[i])],
                                       score=float(scores[i]))
                            for i in range(len(boxes))]
                if self.drop_figure_wrappers:
                    page_dets = drop_redundant_figure_wrappers(
                        page_dets, self.wrapper_parent_cls, self.wrapper_containment_thr,
                        self.wrapper_min_children, self.wrapper_area_ratio_thr)
                out.append(page_dets)
        return out


def layout_prior_from_heron(dets: List[LayoutBox], page_wh: Tuple[float, float]) -> np.ndarray:
    """Vector layout-prior 24-d từ detection Heron, cùng khuôn dạng với
    layout_prior.extract_layout_prior() để trang scan chia sẻ không gian
    cluster với trang PDF born-digital. font_stats [18:24] luôn = 0 (xem
    giới hạn ghi ở đầu file).
    """
    pw, ph = page_wh
    page_area = max(pw * ph, 1e-6)

    text_boxes = [d for d in dets if d.cls in _TEXT_LIKE]
    image_boxes = [d for d in dets if d.cls == "Picture"]
    table_boxes = [d for d in dets if d.cls == "Table"]

    def area(b: LayoutBox) -> float:
        x1, y1, x2, y2 = b.box
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    text_area = sum(area(b) for b in text_boxes)
    image_area = sum(area(b) for b in image_boxes)
    text_area_ratio = text_area / page_area
    image_area_ratio = image_area / page_area
    whitespace_ratio = max(0.0, 1.0 - text_area_ratio - image_area_ratio)
    aspect = float(pw / ph) if ph else 0.0
    block_density = len(text_boxes) / (page_area / 1e6)
    avg_lines_per_block = 0.0   # Heron không cho số dòng/khối (khác PyMuPDF) -> 0

    x_centers = np.array([(b.box[0] + b.box[2]) / 2.0 for b in text_boxes])
    n_cols = _estimate_columns(x_centers, pw)
    col_onehot = np.zeros(6, np.float32)
    col_onehot[min(n_cols, 6) - 1] = 1.0

    n_blocks = len(text_boxes) + len(image_boxes)
    n_text = max(1, len(text_boxes))
    content_hist = np.array([
        len(image_boxes) / n_blocks if n_blocks else 0.0,
        len(table_boxes) / n_text,                          # Table có class riêng -> khỏi đoán heuristic
        (len(text_boxes) - len(table_boxes)) / n_text if len(text_boxes) else 0.0,
        1.0 if not dets else 0.0,                            # trang trắng
        whitespace_ratio,
        image_area_ratio,
    ], np.float32)

    density_block = np.array([
        text_area_ratio, image_area_ratio, whitespace_ratio, aspect,
        block_density, avg_lines_per_block,
    ], np.float32)

    font_stats = np.zeros(6, np.float32)   # không có, xem docstring

    vec = np.concatenate([density_block, col_onehot, content_hist, font_stats]).astype(np.float32)
    assert vec.shape[0] == LAYOUT_DIM, vec.shape
    return vec
