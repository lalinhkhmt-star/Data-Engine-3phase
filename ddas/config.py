"""Cấu hình DDAS (Diversity-and-Difficulty-Aware Sampling) — Phần 1 Data Engine."""
from dataclasses import dataclass, field
from typing import Dict, Tuple

SUBTASKS = ("layout", "text", "formula", "table")


@dataclass
class EmbedConfig:
    """Biểu diễn trang: visual + layout-prior (tuỳ chọn).

    Hai mặc định dưới đây ĐÃ ĐO, không phải phỏng đoán — xem
    data/EMBEDDING_EVAL_REPORT.md và data/results_*.json (300 trang DocLayNet,
    GPU thật). NMI phân cụm theo 6 loại tài liệu:

        model              vit    layout   vit+layout
        CLIP-B/32         0.468    0.151      0.359     <- ghép làm TỆ ĐI
        DINOv2-base       0.196    0.151      0.239     <- ghép giúp
        DiT-base          0.211    0.151      0.143
        DINOv2-small      0.155    0.151      0.159

    Đọc ra hai điều: (1) CLIP hơn DINOv2-base 2.4 lần, nên đổi encoder;
    (2) layout-prior chỉ cứu được encoder YẾU — với encoder mạnh nó chỉ pha
    loãng tín hiệu, làm rơi NMI 23%. Nên layout_weight mặc định = 0.
    CẢNH BÁO: số trên đo ở trang born-digital DocLayNet (tiếng Anh). Với pool
    scan, layout-prior đến từ Heron chứ không phải text layer PDF, chưa đo —
    nếu muốn bật lại thì đo trước, đừng chỉnh mù.
    """
    vit_name: str = "openai/clip-vit-base-patch32"   # đo được NMI 0.468, hơn DINOv2-base (0.196)
    pca_dim: int = 512                           # -> 512-d như paper
    layout_dim: int = 24                         # histogram loại element + mật độ + #cột
    visual_weight: float = 1.0
    layout_weight: float = 0.0                   # xem bảng trên: ghép vào làm tệ đi với encoder mạnh
    # Pool mục tiêu là 100% tài liệu SCAN nên không còn nguồn 'pdf' (text layer
    # PyMuPDF) — trang scan không có text layer, layout_prior.py trả vector ~0.
    # Chỉ còn Heron (RT-DETRv2, GPU) — xem layout_heron.py.
    layout_source_order: Tuple[str, ...] = ("docling-layout-heron",)
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
class JudgeRefineConfig:
    """§3.3 — vòng lặp render-then-verify + hàng đợi chú thích thủ công."""
    max_rounds: int = 3             # paper không ghi số; 3 vòng là điểm cân bằng chi phí/độ hồi phục
    dpi: int = 150
    latex_backends: Tuple[str, ...] = ("pdflatex", "mathtext")
    # Subtask có đường render (formula->LaTeX, table->HTML). text vẫn chạy được
    # vòng judge nhưng chỉ so với ẢNH GỐC (không có ảnh render để đối chiếu);
    # layout không có chuỗi nháp nên đi thẳng sang người.
    renderable: Tuple[str, ...] = ("formula", "table")
    judgeable: Tuple[str, ...] = ("formula", "table", "text")
    # Vòng sau sửa gần như không khác vòng trước mà VẪN báo lỗi => refine bí,
    # dừng sớm thay vì đốt thêm lời gọi model.
    converge_tau: float = 0.995
    # Ngưỡng coi là "judge khoanh được lỗi một cách chắc chắn" — dùng cho tiêu
    # chí ưu tiên #1 (correction efficiency) khi xếp hàng đợi người.
    min_confidence: float = 0.70
    expert_budget: int = 192_000    # số mẫu chú thích tay, theo paper (dòng 66)
    # TRẦN số mẫu Hard được đưa vào vòng Judge-and-Refine. KHÔNG có trong paper
    # nhưng bắt buộc phải có: paper nói §3.3 xử lý "Hard samples" mà không nói
    # bao nhiêu, còn tổng kết dòng 66 chỉ nhắc 192K mẫu chú thích tay. Chạy
    # đúng chữ (toàn bộ Hard) thì ở quy mô 60M trang x 30 element x ~12% Hard
    # = 216M mẫu, nhân 2 vòng x 2 ảnh => ~$22M chỉ riêng tiền gọi trọng tài,
    # gấp ~80 lần toàn bộ §3.1+§3.2 (xem costmodel.judge_refine_plan).
    # Đặt trần rồi chọn mẫu theo ưu tiên: cần `expert_budget` mẫu ra người, với
    # resolve_rate ~45% thì đưa vào khoảng expert_budget/(1-resolve_rate) là đủ,
    # cộng hệ số an toàn. Phần Hard vượt trần KHÔNG bị vứt — nằm lại hàng đợi,
    # chạy đợt sau khi còn ngân sách.
    judge_budget: int = 400_000


@dataclass
class DDASConfig:
    embed: EmbedConfig = field(default_factory=EmbedConfig)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    cmcv: CMCVConfig = field(default_factory=CMCVConfig)
    probe: ProbeConfig = field(default_factory=ProbeConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    judge: JudgeRefineConfig = field(default_factory=JudgeRefineConfig)
    pool_size: int = 500_000_000
    page_budget: int = 60_000_000
