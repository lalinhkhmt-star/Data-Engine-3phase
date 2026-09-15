"""Nhúng trang BẰNG MODEL THẬT — thay thế EmbedConfig mồ côi trong config.py.

Đây là phần trước đây chưa tồn tại: pipeline.py nhận X như tham số có sẵn,
không có code nào tạo ra X từ ảnh trang thật. Module này lấp chỗ đó.

Hai nguồn đặc trưng, ghép lại theo đúng nghi ngờ đã nêu (DINOv2 pretrain trên
ảnh tự nhiên, không đảm bảo tách được cấu trúc bố cục tài liệu):

  1. ViTPageEncoder   — CLS token của ViT-base (mặc định DINOv2), nắm texture/
                        mật độ pixel tổng thể: chữ viết tay, chất lượng scan,
                        script ngôn ngữ.
  2. LayoutPriorExtractor — vector hình học từ PyMuPDF (số cột, mật độ khối,
                        histogram loại nội dung...), nắm cấu trúc bố cục mà
                        ViT không có prior để học.

`eval_embedding.py` trong cùng thư mục đo bằng dữ liệu thật xem ghép hai
nguồn này có thực sự tốt hơn dùng riêng từng cái không — đừng tin cấu hình
mặc định, hãy chạy đánh giá trước.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
from PIL import Image


@dataclass
class ViTPageEncoder:
    """Bọc HuggingFace AutoModel cho một ViT-base bất kỳ (DINOv2, CLIP, ...).

    API tối thiểu: encode(list[PIL.Image]) -> (N, d) float32, đã lấy CLS token.
    Không chuẩn hoá — chuẩn hoá là việc của người gọi (xem `combine_features`),
    vì các candidate embedding khác nhau cần chuẩn hoá khác nhau khi so sánh.
    """
    model_name: str = "facebook/dinov2-base"
    device: str = "cpu"
    batch_size: int = 16
    _processor: object = None
    _model: object = None

    def _lazy_load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoImageProcessor, AutoModel
        self._processor = AutoImageProcessor.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name).eval().to(self.device)
        self._torch = torch

    @property
    def output_dim(self) -> int:
        self._lazy_load()
        return self._model.config.hidden_size

    def encode(self, images: Sequence[Image.Image]) -> np.ndarray:
        self._lazy_load()
        out = []
        for i in range(0, len(images), self.batch_size):
            batch = list(images[i:i + self.batch_size])
            inputs = self._processor(images=batch, return_tensors="pt").to(self.device)
            with self._torch.no_grad():
                o = self._model(**inputs)
            out.append(o.last_hidden_state[:, 0, :].cpu().numpy())  # CLS token
        return np.concatenate(out, axis=0).astype(np.float32)


def render_pdf_page(path: str, page_no: int = 0, dpi: int = 150) -> Image.Image:
    """Render một trang PDF thành PIL.Image bằng PyMuPDF."""
    import fitz  # PyMuPDF
    doc = fitz.open(path)
    page = doc[page_no]
    zoom = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    doc.close()
    return img


def l2_normalize(X: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.clip(n, eps, None)


def fit_pca(X: np.ndarray, dim: int, seed: int = 0) -> np.ndarray:
    """PCA tối giản bằng SVD — đủ dùng cho batch đánh giá, không cần sklearn.

    LƯU Ý: PCA fit trên chính tập đánh giá. Ở quy mô sản xuất phải fit trên
    một mẫu lớn cố định rồi áp dụng lại (không fit theo từng batch), nếu không
    hai lần chạy trên tập khác nhau sẽ có không gian embedding khác nhau và
    không so sánh được.
    """
    mean = X.mean(axis=0, keepdims=True)
    Xc = X - mean
    rng = np.random.default_rng(seed)
    # randomized SVD đơn giản cho ổn định số khi N lớn hơn d nhiều
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:dim].T


def combine_features(vit: Optional[np.ndarray], layout: Optional[np.ndarray],
                     vit_weight: float = 1.0, layout_weight: float = 0.6) -> np.ndarray:
    """Ghép hai nguồn đặc trưng, MỖI NGUỒN CHUẨN HOÁ ĐỘC LẬP trước khi ghép.

    Nếu không chuẩn hoá riêng, nguồn có norm lớn hơn (thường là ViT vì chiều
    cao hơn nhiều) sẽ áp đảo khoảng cách Euclidean trong K-Means bất kể trọng
    số đặt ra sao — đây chính là lỗi "quên L2-normalize" đã nêu ở lượt trước.
    """
    parts = []
    if vit is not None:
        parts.append(vit_weight * l2_normalize(vit))
    if layout is not None:
        parts.append(layout_weight * l2_normalize(layout))
    if not parts:
        raise ValueError("cần ít nhất một nguồn đặc trưng (vit hoặc layout)")
    return np.concatenate(parts, axis=1).astype(np.float32)
