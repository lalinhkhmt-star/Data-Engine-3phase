"""Đặc trưng hình học của trang, tính trực tiếp từ text layer PDF (PyMuPDF).

Đây là nguồn đặc trưng thứ hai để so sánh với ViT-base trong eval_embedding.py.
Không cần GPU, không cần model — chỉ đọc bbox của các khối trong PDF born-digital.
Với trang scan (không có text layer), toàn vector này gần như bằng 0 (được
đánh dấu qua is_valid=False) — hạn chế thật, không che giấu.

24 chiều theo đúng layout_dim trong config.py, chia làm 4 nhóm:
  [0:6]   mật độ & tỉ lệ trang
  [6:12]  cấu trúc cột (ước lượng số cột bằng phân cụm tâm-x các khối)
  [12:18] histogram loại nội dung (chữ / ảnh / bảng-nghi-vấn / trống)
  [18:24] thống kê font (cỡ chữ trung bình, độ lệch — phân biệt heading-dày
          đặc vs văn bản thường)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


LAYOUT_DIM = 24


@dataclass
class LayoutPrior:
    vector: np.ndarray       # (24,) float32
    is_valid: bool           # False nếu trang không có text layer (scan thuần)
    n_blocks: int
    n_columns_est: int


def _estimate_columns(x_centers: np.ndarray, page_w: float, gap_frac: float = 0.04) -> int:
    """Ước lượng số cột: histogram tâm-x các khối, đếm cụm tách biệt bởi khoảng trống.

    Không dùng K-Means (số cột nhỏ, rời rạc) — chỉ cần gom các đỉnh histogram
    liền kề nếu khoảng cách giữa chúng nhỏ hơn gap_frac * bề rộng trang.
    """
    if len(x_centers) < 2:
        return 1
    xs = np.sort(x_centers)
    gaps = np.diff(xs)
    thr = gap_frac * page_w
    n_splits = int((gaps > thr).sum())
    return min(n_splits + 1, 6)


def extract_layout_prior(pdf_path: str, page_no: int = 0) -> LayoutPrior:
    import fitz  # PyMuPDF

    doc = fitz.open(pdf_path)
    page = doc[page_no]
    pw, ph = page.rect.width, page.rect.height
    d = page.get_text("dict")
    doc.close()

    blocks = d.get("blocks", [])
    text_blocks, image_blocks = [], []
    font_sizes: list[float] = []
    x_centers: list[float] = []
    areas: list[float] = []
    line_counts: list[int] = []

    for b in blocks:
        x0, y0, x1, y1 = b["bbox"]
        area = max(0.0, (x1 - x0)) * max(0.0, (y1 - y0))
        if b.get("type", 0) == 1:                       # ảnh
            image_blocks.append(area)
            continue
        lines = b.get("lines", [])
        if not lines:
            continue
        text_blocks.append(area)
        x_centers.append((x0 + x1) / 2.0)
        areas.append(area)
        line_counts.append(len(lines))
        for ln in lines:
            for sp in ln.get("spans", []):
                if sp.get("size"):
                    font_sizes.append(sp["size"])

    n_blocks = len(text_blocks) + len(image_blocks)
    is_valid = len(text_blocks) > 0 and pw > 0 and ph > 0

    if not is_valid:
        return LayoutPrior(np.zeros(LAYOUT_DIM, np.float32), False, n_blocks, 0)

    page_area = pw * ph
    text_area_ratio = float(np.sum(areas) / page_area) if areas else 0.0
    image_area_ratio = float(np.sum(image_blocks) / page_area) if image_blocks else 0.0
    whitespace_ratio = max(0.0, 1.0 - text_area_ratio - image_area_ratio)
    aspect = float(pw / ph) if ph else 0.0
    block_density = len(text_blocks) / (page_area / 1e6)   # khối / triệu-pixel²
    avg_lines_per_block = float(np.mean(line_counts)) if line_counts else 0.0

    n_cols = _estimate_columns(np.array(x_centers), pw)
    col_onehot = np.zeros(6, np.float32)
    col_onehot[min(n_cols, 6) - 1] = 1.0

    # histogram loại nội dung nghi vấn theo hình dạng khối:
    #   khối rất rộng & thấp, nhiều dòng ngắn đều nhau -> nghi bảng
    #   khối cao hẹp -> nghi caption/sidebar
    #   còn lại -> văn bản thường
    table_like = sum(1 for a, lc in zip(areas, line_counts)
                     if lc >= 3 and a > 0 and (a / page_area) > 0.03
                     and lc / max(1, np.sqrt(a) / 20) > 0.15)
    n_text = max(1, len(text_blocks))
    content_hist = np.array([
        len(image_blocks) / n_blocks if n_blocks else 0.0,
        table_like / n_text,
        (n_text - table_like) / n_text,
        1.0 if not text_blocks and not image_blocks else 0.0,   # trang trắng
        whitespace_ratio,
        image_area_ratio,
    ], np.float32)

    fs = np.array(font_sizes, np.float32) if font_sizes else np.array([0.0], np.float32)
    font_stats = np.array([
        float(fs.mean()), float(fs.std()), float(fs.min()), float(fs.max()),
        float(np.percentile(fs, 90)), float((fs > fs.mean() + fs.std()).mean()),  # tỉ lệ "heading"
    ], np.float32)

    density_block = np.array([
        text_area_ratio, image_area_ratio, whitespace_ratio, aspect,
        block_density, avg_lines_per_block,
    ], np.float32)

    vec = np.concatenate([density_block, col_onehot, content_hist, font_stats]).astype(np.float32)
    assert vec.shape[0] == LAYOUT_DIM, vec.shape
    return LayoutPrior(vec, True, n_blocks, n_cols)
