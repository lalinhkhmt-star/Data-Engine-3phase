"""Cấu hình DDAS (Diversity-and-Difficulty-Aware Sampling) — Phần 1 Data Engine."""
from dataclasses import dataclass, field
from typing import Dict, Tuple

SUBTASKS = ("layout", "text", "formula", "table")


@dataclass
class EmbedConfig:
    """Biểu diễn trang: visual (ViT-base) + layout-prior (task-aware)."""
    vit_name: str = "facebook/dinov2-base"      # 768-d CLS
    pca_dim: int = 512                           # -> 512-d như paper
    layout_dim: int = 24                         # histogram loại element + mật độ + #cột
    visual_weight: float = 1.0
    layout_weight: float = 0.6                   # task-aware: tách trang theo cấu trúc, không chỉ "nhìn giống nhau"
    # Nguồn layout-prior: 'pdf' (PyMuPDF, CPU, ~free) cho born-digital, 'yolo' cho trang scan.
    layout_source_order: Tuple[str, ...] = ("pdf", "yolo")
    batch_size: int = 512
    fp16: bool = True


@dataclass
class ClusterConfig:
    """K-Means phân cấp cân bằng + kênh tail riêng."""
    k_coarse: int = 512
    split_factor: int = 16          # cluster > max_share sẽ bị chẻ tiếp
    max_share: float = 0.005        # 0.5% pool -> cluster "đầu", phải chẻ
    min_cluster_size: int = 200
    max_depth: int = 3
    fit_sample: int = 20_000_000    # số trang dùng để fit centroid
    kmeans_iters: int = 25
    tail_percentile: float = 99.5   # d(x, centroid gần nhất) > p99.5 -> tail reservoir
    dedup_cosine: float = 0.98      # khử near-duplicate trong cụm trước khi lấy mẫu
    # Làm phẳng mật độ trước khi fit centroid (xem cluster.py::_kmeans_flattened).
    # K-Means chuẩn đặt centroid TỈ LỆ VỚI MẬT ĐỘ, nên vùng "đầu" chiếm phần lớn
    # số cụm và mọi cách chia đều theo cụm vẫn kế thừa nguyên long-tail shift.
    flatten_rounds: int = 5
    flatten_fit_cap: int = 2_000_000   # số điểm resample mỗi vòng làm phẳng


@dataclass
class CMCVConfig:
    """Ngưỡng đồng thuận cho từng subtask; hiệu chuẩn trên dev-set có GT."""
    # tau: điểm tương đồng >= tau  =>  coi là "đồng thuận"
    tau: Dict[str, float] = field(default_factory=lambda: {
        "text": 0.92,      # 1 - NED
        "table": 0.95,     # TEDS
        "formula": 0.95,   # CDM
        "layout": 0.85,    # element-set IoU/F1
    })
    # Mục tiêu hiệu chuẩn: P(đúng | đồng thuận) >= precision_target trên dev-set
    precision_target: float = 0.98
    cascade: bool = True            # bỏ qua model 30B khi hai model rẻ đã đồng thuận
    require_3way_for_easy: bool = False  # bật nếu dev-set cho thấy lỗi tương quan


@dataclass
class ProbeConfig:
    """Probe-and-extrapolate: chỉ chạy CMCV trên mẫu dò của mỗi cluster."""
    n_min: int = 64
    n_max: int = 512
    beta: float = 0.35              # n0 = clip(beta*sqrt(N_c), n_min, n_max)
    invalid_drop: float = 0.50      # cluster có >50% invalid -> loại
    prior_strength: float = 20.0    # empirical-Bayes shrink về prior toàn cục
    # Gain theo độ khó (giá trị huấn luyện biên), theo Section 3.2
    gain: Dict[str, Tuple[float, float, float]] = field(default_factory=lambda: {
        #          (Easy, Medium, Hard)
        "layout":  (0.15, 1.00, 0.60),
        "text":    (0.10, 1.00, 0.50),   # text hưởng lợi nhiều từ Medium
        "formula": (0.10, 0.80, 1.00),   # formula/table nhạy với Hard
        "table":   (0.10, 0.80, 1.00),
    })
    entropy_bonus: float = 0.35     # cluster có phân bố độ khó đa dạng -> weight cao hơn


@dataclass
class SamplerConfig:
    alpha: float = 0.4              # nhiệt độ long-tail: quota ~ N^alpha (0=đều, 1=tỉ lệ)
    floor_per_cell: int = 8         # đảm bảo phủ cluster hiếm
    cap_ratio: float = 0.60         # không lấy quá 60% một cell (tránh over-fit cluster nhỏ)
    budget: Dict[str, int] = field(default_factory=lambda: {
        "layout": 20_000_000,
        "text": 25_000_000,
        "formula": 8_000_000,
        "table": 7_000_000,
    })
    waterfill_iters: int = 50


@dataclass
class DDASConfig:
    embed: EmbedConfig = field(default_factory=EmbedConfig)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    cmcv: CMCVConfig = field(default_factory=CMCVConfig)
    probe: ProbeConfig = field(default_factory=ProbeConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    pool_size: int = 500_000_000
    page_budget: int = 60_000_000
