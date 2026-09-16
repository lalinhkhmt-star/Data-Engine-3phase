"""Mô hình chi phí — cơ sở định lượng cho phần "tính khả thi".

Pipeline giờ trộn 2 loại chi phí khác bản chất:
  - 2/3 model CMCV TỰ HOST trên GPU thuê (Qwen3-VL target + PaddleOCR-VL
    trọng tài) -> chi phí = GPU-giờ x giá thuê. Đơn giá lấy theo throughput
    đo được thực tế trên A100-80G (fp16, vLLM batching, ảnh 200 dpi ~1600px
    cạnh dài) khi có số công bố; nếu không thì ước và đánh dấu rõ. Chỉnh
    `RATE` theo cụm máy thật rồi chạy lại.
  - Mistral OCR gọi qua API nhà cung cấp -> chi phí = số trang x giá mỗi
    trang, KHÔNG tốn GPU/wall-clock của bạn nhưng vẫn tốn tiền thật và bị
    giới hạn bởi rate-limit của nhà cung cấp (không mô hình hoá ở đây). Giá
    trong `API_COST_PER_PAGE_USD` lấy từ trang giá công bố ~09/2026 — ĐÂY LÀ
    SỐ THAM KHẢO, kiểm tra lại giá hiện hành trước khi dùng để quyết định
    ngân sách thật.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

# pages (hoặc items) mỗi giây trên MỘT GPU A100-80G — chỉ áp dụng cho stage kind="gpu"
RATE = {
    "vit_base_page":     900.0,    # DINOv2-B, 224px, batch 512
    "vit_small_crop":   3200.0,    # element crop
    # Docling Layout Heron (RT-DETRv2) — chỉ dùng cho trang scan, xem layout_heron.py.
    # CHƯA ĐO THẬT: ước từ Heron-101 (bản rút gọn, README công bố ~28ms/trang trên
    # 1 A100 => ~36/s) chia đôi vì model Heron đầy đủ lớn hơn, chưa có số công bố.
    "docling-layout-heron": 18.0,
    "pdf_layout_cpu":    200.0,    # PyMuPDF, mỗi CPU-core (born-digital: miễn phí GPU)
    "qwen3-vl-8b":         1.0,    # target — 8B dense VLM, CHƯA ĐO THẬT, ước theo tỉ lệ tham số
    # PaddleOCR-VL (0.9B, ERNIE-4.5 + visual encoder) — trọng tài CMCV, TỰ HOST.
    # ĐO THẬT, không phải ước lượng: 1.224 trang/s trên 1 A100 (backend cơ bản,
    # arXiv 2510.14528); FastDeploy backend nhanh hơn (~1.618/s trên A100,
    # ~2.225/s trên H800) — dùng số cơ bản cho thận trọng, đổi khi đo trên
    # cụm máy thật. Nhanh hơn cả target (8B) vì model chỉ 0.9B tham số.
    "paddleocr-vl":     1.224,
}
GPU_HOUR_USD = 1.8                 # giá thuê tham chiếu

# USD mỗi trang qua API — chỉ áp dụng cho stage kind="api". Nguồn: trang giá
# công bố của Mistral ~09/2026, giá đồng bộ (chưa trừ batch discount).
API_COST_PER_PAGE_USD = {
    # Mistral OCR 4: $4/1000 trang đồng bộ, $2/1000 trang qua Batch API (-50%).
    "mistral-ocr-4":   0.004,
    # KHÔNG còn model thứ 3 nào gọi API trong CMCV — PaddleOCR-VL
    # (EXPENSIVE_EXTERNAL) tự host, tính phí qua RATE["paddleocr-vl"] ở trên.
    #
    # Ba mục dưới đây CHỈ dùng cho §3.3 (judge_refine_plan), không nằm trong
    # ddas_plan(). Ước theo cùng cách với CMCV (~1600 token ảnh+prompt vào,
    # ~700 token ra mỗi ảnh) — CHƯA ĐO TRÊN DỮ LIỆU THẬT.
    "judge-vlm":       0.026,   # GPT-5 lớp trọng tài; 2 ảnh/lời gọi đã nhân ở stage
    "gemini-3-pro":    0.012,   # pre-annotation, $2/$12 mỗi triệu token in/out
    "expert-human":    1.20,    # tiền thuê người mỗi mẫu — con số phải thương lượng thật
}


@dataclass
class Stage:
    name: str
    items: float
    rate_key: str
    note: str = ""
    kind: str = "gpu"          # "gpu" (tự host, dùng RATE) | "api" (gọi ngoài, dùng API_COST_PER_PAGE_USD)

    @property
    def gpu_hours(self) -> float:
        if self.kind != "gpu":
            return 0.0
        return self.items / RATE[self.rate_key] / 3600.0

    @property
    def usd(self) -> float:
        if self.kind == "api":
            return self.items * API_COST_PER_PAGE_USD[self.rate_key]
        return self.gpu_hours * GPU_HOUR_USD


def ddas_plan(pool: float = 500e6, page_budget: float = 60e6,
              k_clusters: int = 4096, n_probe: int = 256,
              born_digital_frac: float = 0.75,
              easy_rate: float = 0.60,
              elements_per_page: float = 30.0,
              cascade: bool = True) -> List[Stage]:
    scanned = pool * (1 - born_digital_frac)
    probe_pages = k_clusters * n_probe
    # cascade: chỉ gọi model thứ 3 (PaddleOCR-VL, tự host) trên phần target và
    # Mistral OCR bất đồng — tiết kiệm GPU-giờ, không phải tiết kiệm $ API.
    q_frac = (1.0 - easy_rate) if cascade else 1.0
    return [
        Stage("1a  Nhúng trang (ViT-base)", pool, "vit_base_page", "toàn pool"),
        Stage("1a  Layout prior (scan only)", scanned, "docling-layout-heron",
              f"{100*(1-born_digital_frac):.0f}% pool; born-digital dùng PyMuPDF trên CPU"),
        Stage("1b  Probe CMCV · Qwen3-VL", probe_pages, "qwen3-vl-8b", f"{k_clusters}x{n_probe}"),
        Stage("1b  Probe CMCV · Mistral OCR", probe_pages, "mistral-ocr-4", "", kind="api"),
        Stage("1b  Probe CMCV · PaddleOCR-VL", probe_pages * q_frac, "paddleocr-vl",
              f"cascade cắt {100*easy_rate:.0f}%" if cascade else "no cascade"),
        Stage("2   CMCV đầy đủ · Qwen3-VL", page_budget, "qwen3-vl-8b", "cũng chính là nhãn Easy"),
        Stage("2   CMCV đầy đủ · Mistral OCR", page_budget, "mistral-ocr-4", "", kind="api"),
        Stage("2   CMCV đầy đủ · PaddleOCR-VL", page_budget * q_frac, "paddleocr-vl",
              f"cascade cắt {100*easy_rate:.0f}%" if cascade else "no cascade"),
        Stage("2b  Layout detect (Heron) · candidate set", page_budget, "docling-layout-heron",
              "chạy TRÊN TOÀN candidate set (không chỉ trang scan) — suy luận thật cho Stage 2, xem element.py"),
        Stage("2   Nhúng element", page_budget * elements_per_page, "vit_small_crop",
              f"{elements_per_page:.0f} element/trang"),
    ]


def judge_refine_plan(judge_budget: float = 400e3,
                      mean_rounds: float = 2.0,
                      resolve_rate: float = 0.45,
                      expert_budget: float = 192e3,
                      usd_per_expert_item: float = 1.20,
                      page_budget: Optional[float] = None,
                      elements_per_page: float = 30.0,
                      hard_rate: float = 0.12) -> List[Stage]:
    """Chi phí §3.3 — phần `ddas_plan()` KHÔNG tính (nó dừng ở stage 2).

    `judge_budget` là TRẦN số mẫu Hard được đưa vào vòng, khớp với
    JudgeRefineConfig.judge_budget. Truyền `page_budget` để thay vào đó tính
    theo kịch bản "chạy hết Hard như paper viết" — dùng để thấy vì sao phải có
    trần: ở 60M trang, cách đó ra ~216M mẫu và ~$22M tiền gọi trọng tài.

    MỌI tham số dưới đây CHƯA ĐO, vì chưa gọi model trọng tài thật lần nào:
      mean_rounds  số vòng judge trung bình mỗi mẫu (cfg.max_rounds=3 là trần)
      resolve_rate tỉ lệ Hard tầng tự động cứu được
      usd_per_expert_item  đơn giá thuê người mỗi mẫu — phụ thuộc thị trường
                   lao động và độ khó, chênh nhau nhiều lần giữa các nơi.
    Chạy `JudgeRefine.stats` trên vài trăm mẫu thật rồi thay số vào đây.

    Ảnh gốc + ảnh render là HAI ảnh mỗi lời gọi nên token vào gấp đôi so với
    CMCV — phản ánh bằng cách tính mỗi vòng là 2 "trang" theo đơn giá API.
    """
    hard_items = (page_budget * elements_per_page * hard_rate
                  if page_budget is not None else judge_budget)
    judge_calls = hard_items * mean_rounds
    preannot_items = hard_items * (1.0 - resolve_rate)
    return [
        Stage("3.3 Judge-and-Refine (GPT-5)", judge_calls * 2, "judge-vlm",
              f"{hard_items/1e6:.1f}M mẫu Hard x {mean_rounds:.1f} vòng, 2 ảnh/lời gọi",
              kind="api"),
        Stage("3.3 Pre-annotation (Gemini 3 Pro)", preannot_items, "gemini-3-pro",
              f"{100*(1-resolve_rate):.0f}% Hard không tự cứu được", kind="api"),
        Stage("3.3 Chú thích tay (chuyên gia)", expert_budget, "expert-human",
              f"${usd_per_expert_item:.2f}/mẫu — CHƯA ĐO, phụ thuộc thị trường", kind="api"),
    ]


def summarize(stages: List[Stage], n_gpu: int = 256) -> Dict[str, float]:
    gh = sum(s.gpu_hours for s in stages)
    usd = sum(s.usd for s in stages)
    api_usd = sum(s.usd for s in stages if s.kind == "api")
    return {"gpu_hours": gh, "gpu_days": gh / 24,
            "wall_days": gh / n_gpu / 24, "usd": usd, "api_usd": api_usd}


def render(stages: List[Stage], n_gpu: int = 256) -> str:
    w = max(len(s.name) for s in stages) + 2
    lines = [f"{'Giai đoạn'.ljust(w)}{'Items':>14}{'GPU-h':>11}{'USD':>12}   Ghi chú",
             "-" * (w + 28 + 12 + 40)]
    for s in stages:
        lines.append(f"{s.name.ljust(w)}{s.items:>14,.0f}{s.gpu_hours:>11,.0f}"
                     f"{s.usd:>12,.0f}   {s.note}")
    t = summarize(stages, n_gpu)
    lines += ["-" * (w + 28 + 12 + 40),
              f"{'TỔNG'.ljust(w)}{'':>14}{t['gpu_hours']:>11,.0f}{t['usd']:>12,.0f}   "
              f"= {t['gpu_days']:,.0f} GPU-ngày (tự host) · {t['wall_days']:.1f} ngày trên {n_gpu} GPU "
              f"· trong đó chi phí API ~${t['api_usd']:,.0f} · tổng ~${t['usd']:,.0f}"]
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
