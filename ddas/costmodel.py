"""Mô hình chi phí — cơ sở định lượng cho phần "tính khả thi".

Đơn giá lấy theo throughput đo được thực tế trên A100-80G (fp16, vLLM batching,
ảnh 200 dpi ~1600px cạnh dài). Chỉnh `RATE` theo cụm máy thật rồi chạy lại.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

# pages (hoặc items) mỗi giây trên MỘT GPU A100-80G
RATE = {
    "vit_base_page":     900.0,    # DINOv2-B, 224px, batch 512
    "vit_small_crop":   3200.0,    # element crop
    "doclayout_yolo":    110.0,    # chỉ dùng cho trang scan
    "pdf_layout_cpu":    200.0,    # PyMuPDF, mỗi CPU-core (born-digital: miễn phí GPU)
    "mineru2.5":           2.2,    # 1.2B VLM
    "paddleocr-vl":        2.4,    # 0.9B VLM
    "qwen3-vl-30b":        0.32,   # 30B-A3B MoE
}
GPU_HOUR_USD = 1.8                 # giá thuê tham chiếu


@dataclass
class Stage:
    name: str
    items: float
    rate_key: str
    note: str = ""

    @property
    def gpu_hours(self) -> float:
        return self.items / RATE[self.rate_key] / 3600.0


def ddas_plan(pool: float = 500e6, page_budget: float = 60e6,
              k_clusters: int = 4096, n_probe: int = 256,
              born_digital_frac: float = 0.75,
              easy_rate: float = 0.60,
              elements_per_page: float = 30.0,
              cascade: bool = True) -> List[Stage]:
    scanned = pool * (1 - born_digital_frac)
    probe_pages = k_clusters * n_probe
    # cascade: chỉ gọi model 30B trên phần MinerU và Paddle bất đồng
    q_frac = (1.0 - easy_rate) if cascade else 1.0
    return [
        Stage("1a  Nhúng trang (ViT-base)", pool, "vit_base_page", "toàn pool"),
        Stage("1a  Layout prior (scan only)", scanned, "doclayout_yolo",
              f"{100*(1-born_digital_frac):.0f}% pool; born-digital dùng PyMuPDF trên CPU"),
        Stage("1b  Probe CMCV · MinerU", probe_pages, "mineru2.5", f"{k_clusters}x{n_probe}"),
        Stage("1b  Probe CMCV · Paddle", probe_pages, "paddleocr-vl", ""),
        Stage("1b  Probe CMCV · Qwen30B", probe_pages * q_frac, "qwen3-vl-30b",
              f"cascade cắt {100*easy_rate:.0f}%" if cascade else "no cascade"),
        Stage("2   CMCV đầy đủ · MinerU", page_budget, "mineru2.5", "cũng chính là nhãn Easy"),
        Stage("2   CMCV đầy đủ · Paddle", page_budget, "paddleocr-vl", ""),
        Stage("2   CMCV đầy đủ · Qwen30B", page_budget * q_frac, "qwen3-vl-30b",
              f"cascade cắt {100*easy_rate:.0f}%" if cascade else "no cascade"),
        Stage("2   Nhúng element", page_budget * elements_per_page, "vit_small_crop",
              f"{elements_per_page:.0f} element/trang"),
    ]


def summarize(stages: List[Stage], n_gpu: int = 256) -> Dict[str, float]:
    gh = sum(s.gpu_hours for s in stages)
    return {"gpu_hours": gh, "gpu_days": gh / 24,
            "wall_days": gh / n_gpu / 24, "usd": gh * GPU_HOUR_USD}


def render(stages: List[Stage], n_gpu: int = 256) -> str:
    w = max(len(s.name) for s in stages) + 2
    lines = [f"{'Giai đoạn'.ljust(w)}{'Items':>14}{'GPU-h':>11}   Ghi chú",
             "-" * (w + 28 + 40)]
    for s in stages:
        lines.append(f"{s.name.ljust(w)}{s.items:>14,.0f}{s.gpu_hours:>11,.0f}   {s.note}")
    t = summarize(stages, n_gpu)
    lines += ["-" * (w + 28 + 40),
              f"{'TỔNG'.ljust(w)}{'':>14}{t['gpu_hours']:>11,.0f}   "
              f"= {t['gpu_days']:,.0f} GPU-ngày · {t['wall_days']:.1f} ngày trên {n_gpu} GPU "
              f"· ~${t['usd']:,.0f}"]
    return "\n".join(lines)


def storage_plan(page_budget: float = 60e6, pool: float = 500e6,
                 elements_per_page: float = 30.0) -> Dict[str, float]:
    """Dung lượng (TB)."""
    return {
        "embedding trang (fp16 512-d, toàn pool)": pool * 512 * 2 / 1e12,
        "embedding element (fp16 384-d)": page_budget * elements_per_page * 384 * 2 / 1e12,
        "ảnh trang đã chọn (JPEG q85 200dpi ~300KB)": page_budget * 300e3 / 1e12,
        "đầu ra parse 3 model (~10KB/trang/model)": page_budget * 3 * 10e3 / 1e12,
        "metadata CMCV + cluster (~1KB/trang)": page_budget * 1e3 / 1e12,
    }
