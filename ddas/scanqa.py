"""Cổng chất lượng ảnh scan — chạy TRƯỚC mọi lời gọi model, thuần CPU.

KHÔNG có trong paper. Paper giả định pool tài liệu chung; pool của dự án này
là 100% tài liệu scan, và chuyển sang scan thì "khó" tách làm hai thứ cần đối
xử NGƯỢC nhau:

  - khó vì CẤU TRÚC (bảng lồng nhau, nhiều cột, công thức dày) -> dữ liệu quý,
    đáng chi tiền chú thích.
  - khó vì ẢNH HỎNG (mờ, nghiêng, nhiễu, photocopy nhiều đời, dấu mộc đè chữ)
    -> KHÔNG quý. Model nào cũng đọc sai, nhưng dạy model đọc trang mà người
    còn không đọc nổi thì không học được gì.

CMCV gộp cả hai vào tier Hard, rồi §3.3 đem cả hai đi gọi trọng tài nhiều vòng
và xếp hàng cho người — tức phần ngân sách đắt nhất chảy vào đúng thứ không
đáng chi. Module này cắt loại thứ hai ra trước khi tốn đồng nào.

Cắm vào đâu: `CMCV(validity_fn=...)` (cmcv.py) chạy validity_fn TRƯỚC khi gọi
model và gán thẳng Tier.INVALID; `ProbeConfig.invalid_drop` đã biết loại cả
cụm rác. Khe cắm có sẵn, đây là bộ đo còn thiếu.

Mọi ngưỡng dưới đây CHƯA HIỆU CHUẨN trên dữ liệu thật — đặt rộng tay để chỉ
bắt trang hỏng rõ ràng. Hiệu chuẩn bằng calibrate_thresholds() khi có mẫu đã
gán nhãn đọc-được/không-đọc-được.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


@dataclass
class ScanQAConfig:
    """Ngưỡng loại trang. Đặt rộng: thà giữ nhầm trang xấu còn hơn vứt nhầm
    trang khó-nhưng-đọc-được (thứ chính ta đang đi tìm)."""
    min_blur: float = 40.0        # phương sai Laplacian; thấp = mờ/out-of-focus
    # Cạnh ngắn. 500px ~ A4 quét 60dpi — dưới mức đó chữ thân bài không còn đủ
    # nét cho OCR nào. KHÔNG đặt cao hơn tuỳ tiện: bản thân data/vietnamese_sample
    # là ảnh 640x640 và hoàn toàn đọc được, ngưỡng 700 đã vứt sạch chúng khi thử.
    min_side_px: int = 500
    min_ink: float = 0.004        # tỉ lệ điểm ảnh tối; dưới = trang gần như trắng
    max_ink: float = 0.60         # trên = trang đen kịt (scan hỏng, ảnh âm bản, trang bị bôi)
    min_contrast: float = 25.0    # độ lệch chuẩn mức xám
    max_skew_deg: float = 12.0    # nghiêng hơn mức này thì nên deskew trước, không loại
    sample_px: int = 1_000_000    # thu nhỏ trước khi đo cho nhanh; đủ để đánh giá chất lượng


@dataclass
class ScanQA:
    """Kết quả đo một trang. `drop=True` => Tier.INVALID, không gọi model."""
    page_id: str
    blur: float
    ink: float
    contrast: float
    width: int
    height: int
    skew_deg: float
    reasons: List[str] = field(default_factory=list)

    @property
    def drop(self) -> bool:
        return bool(self.reasons)

    @property
    def needs_deskew(self) -> bool:
        return "nghiêng" in " ".join(self.reasons) or abs(self.skew_deg) > 1.0


# ------------------------------------------------------------ phép đo ------

def _to_gray(img: Image.Image, sample_px: int) -> np.ndarray:
    w, h = img.size
    if w * h > sample_px:                      # thu nhỏ giữ tỉ lệ cho nhanh
        s = (sample_px / (w * h)) ** 0.5
        img = img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BILINEAR)
    return np.asarray(img.convert("L"), dtype=np.float32)


def _laplacian_var(g: np.ndarray) -> float:
    """Phương sai Laplacian — chỉ số mờ kinh điển. Ảnh nét có nhiều cạnh mạnh
    nên phương sai cao; ảnh mờ bị san phẳng nên thấp."""
    if g.shape[0] < 3 or g.shape[1] < 3:
        return 0.0
    lap = (-4.0 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1]
           + g[1:-1, :-2] + g[1:-1, 2:])
    return float(lap.var())


def _ink_ratio(g: np.ndarray) -> float:
    """Tỉ lệ điểm ảnh tối, ngưỡng theo Otsu đơn giản hoá (trung điểm Otsu 2 lớp)."""
    hist, _ = np.histogram(g, bins=256, range=(0, 255))
    total = hist.sum()
    if total == 0:
        return 0.0
    omega = np.cumsum(hist) / total
    mu = np.cumsum(hist * np.arange(256)) / total
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b = np.where(denom > 0, (mu_t * omega - mu) ** 2 / denom, 0.0)
    thr = float(np.argmax(sigma_b))
    return float((g <= thr).mean())


def _skew_deg(g: np.ndarray) -> float:
    """Ước góc nghiêng bằng cách thử xoay và tìm góc có profile hàng "gai"
    nhất (dòng chữ thẳng hàng => tổng mực theo hàng dao động mạnh nhất).

    Rẻ và đủ dùng cho việc gắn cờ; muốn deskew chính xác thì dùng Hough hoặc
    Radon ở bước tiền xử lý, không phải ở đây.
    """
    small = g[:: max(1, g.shape[0] // 400), :: max(1, g.shape[1] // 400)]
    small = 255.0 - small                       # đảo: mực = giá trị cao
    best, best_angle = -1.0, 0.0
    h, w = small.shape
    if h < 8 or w < 8:
        return 0.0
    yy, xx = np.nonzero(small > small.mean() + small.std())
    if len(yy) < 50:
        return 0.0
    for angle in np.arange(-12.0, 12.01, 1.0):
        t = np.deg2rad(angle)
        proj = yy * np.cos(t) - xx * np.sin(t)
        hist, _ = np.histogram(proj, bins=max(8, h))
        score = float(np.var(hist))
        if score > best:
            best, best_angle = score, float(angle)
    return best_angle


def assess(image: Image.Image, page_id: str = "",
           cfg: Optional[ScanQAConfig] = None) -> ScanQA:
    """Đo một trang. Không sửa ảnh, chỉ đo và gắn cờ."""
    cfg = cfg or ScanQAConfig()
    w, h = image.size
    g = _to_gray(image, cfg.sample_px)
    blur = _laplacian_var(g)
    ink = _ink_ratio(g)
    contrast = float(g.std())
    skew = _skew_deg(g)

    reasons: List[str] = []
    if min(w, h) < cfg.min_side_px:
        reasons.append(f"độ phân giải thấp ({w}x{h}, cạnh ngắn < {cfg.min_side_px})")
    if blur < cfg.min_blur:
        reasons.append(f"mờ (Laplacian var {blur:.1f} < {cfg.min_blur})")
    if ink < cfg.min_ink:
        reasons.append(f"gần như trắng (mực {100*ink:.2f}%)")
    elif ink > cfg.max_ink:
        reasons.append(f"đen bất thường (mực {100*ink:.1f}%)")
    if contrast < cfg.min_contrast:
        reasons.append(f"tương phản thấp (std {contrast:.1f} < {cfg.min_contrast})")

    qa = ScanQA(page_id, blur, ink, contrast, w, h, skew, reasons)
    # Nghiêng KHÔNG phải lý do loại — deskew được, và trang nghiêng vẫn có thể
    # là trang quý. Chỉ gắn cờ để bước tiền xử lý xoay lại.
    return qa


# --------------------------------------------------------- cắm vào CMCV ----

def make_validity_fn(image_fn, cfg: Optional[ScanQAConfig] = None,
                     cache: Optional[Dict[str, ScanQA]] = None):
    """Dựng `validity_fn(page_id) -> bool` cho CMCV (xem cmcv.py::CMCV).

    Trả False => trang bị gán Tier.INVALID và KHÔNG model nào được gọi trên nó.
    Truyền `cache` (dict) nếu muốn giữ lại số đo để báo cáo/hiệu chuẩn sau.
    """
    cfg = cfg or ScanQAConfig()

    def validity_fn(page_id: str) -> bool:
        qa = assess(image_fn(page_id), page_id, cfg)
        if cache is not None:
            cache[page_id] = qa
        return not qa.drop

    return validity_fn


def summarize(results: Sequence[ScanQA]) -> Dict[str, object]:
    """Thống kê để biết pool bẩn tới đâu và tiết kiệm được bao nhiêu lời gọi."""
    n = len(results)
    dropped = [r for r in results if r.drop]
    counts: Dict[str, int] = {}
    for r in dropped:
        for reason in r.reasons:
            key = reason.split(" (")[0]
            counts[key] = counts.get(key, 0) + 1
    return {
        "tổng": n,
        "loại": len(dropped),
        "% loại": 100.0 * len(dropped) / max(1, n),
        "cần deskew": sum(1 for r in results if r.needs_deskew),
        "theo lý do": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
    }


def calibrate_thresholds(results: Sequence[ScanQA], readable: Sequence[bool],
                         target_recall: float = 0.99) -> Dict[str, float]:
    """Chọn ngưỡng blur/contrast giữ lại >= `target_recall` số trang ĐỌC ĐƯỢC.

    `readable[i]` = người xác nhận trang i đọc được hay không. Ưu tiên KHÔNG
    vứt nhầm trang tốt (recall cao) hơn là lọc sạch trang rác — vứt nhầm một
    trang khó-nhưng-đọc-được là mất đúng loại dữ liệu đắt nhất.
    """
    ok = np.array(readable, dtype=bool)
    if ok.sum() == 0:
        return {}
    out: Dict[str, float] = {}
    for name, vals in (("min_blur", np.array([r.blur for r in results])),
                       ("min_contrast", np.array([r.contrast for r in results]))):
        good = vals[ok]
        out[name] = float(np.quantile(good, 1.0 - target_recall))
    return out
