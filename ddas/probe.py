"""Stage 1b — Probe-and-Extrapolate: quyết định trọng số cụm với ngân sách CMCV bé.

Ý tưởng tách chi phí: quyết định *lấy mẫu ở đâu* không cần CMCV trên toàn pool.
Chỉ cần ước lượng phân bố độ khó p_c = (E, M, H, Invalid) của MỖI CỤM. Với n0
mẫu dò, sai số chuẩn của mỗi tỉ lệ là sqrt(p(1-p)/n0) <= 0.5/sqrt(n0):
    n0=256 -> SE <= 3.1%   |   n0=512 -> SE <= 2.2%
Đủ chính xác để xếp hạng cụm, trong khi tổng lời gọi CMCV giảm từ O(pool) xuống
O(K * n0)  ~  vài trăm nghìn trang thay vì hàng trăm triệu.

Ước lượng dùng Empirical Bayes (Dirichlet shrink về prior toàn cục) nên cụm nhỏ
không bị nhiễu ngẫu nhiên đẩy lên/xuống quá mức.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence

import numpy as np

from .cmcv import Tier
from .config import ProbeConfig

TIERS: Sequence[Tier] = (Tier.EASY, Tier.MEDIUM, Tier.HARD, Tier.INVALID)


@dataclass
class ClusterStat:
    cid: int
    size: int
    n_probe: int
    counts: np.ndarray        # (4,) đếm theo TIERS
    p: np.ndarray             # (4,) hậu nghiệm sau shrink
    value: float              # V_c — giá trị huấn luyện kỳ vọng
    weight: float             # trọng số lấy mẫu chưa chuẩn hoá
    dropped: bool
    se: float                 # sai số chuẩn lớn nhất của p


def probe_size(n_cluster: int, cfg: ProbeConfig) -> int:
    return int(np.clip(round(cfg.beta * np.sqrt(n_cluster)), cfg.n_min,
                       min(cfg.n_max, n_cluster)))


def _entropy(p: np.ndarray) -> float:
    q = p[:3]                                   # chỉ E/M/H, bỏ Invalid
    q = q / max(q.sum(), 1e-9)
    q = q[q > 0]
    return float(-(q * np.log(q)).sum() / np.log(3))     # chuẩn hoá về [0,1]


def cluster_weights(assign: np.ndarray,
                    probe_tiers: Dict[int, List[Tier]],
                    subtask: str,
                    cfg: ProbeConfig | None = None,
                    alpha: float = 0.4) -> Dict[int, ClusterStat]:
    """probe_tiers: cid -> danh sách tier CMCV của các trang dò trong cụm đó."""
    cfg = cfg or ProbeConfig()
    gE, gM, gH = cfg.gain[subtask]
    gains = np.array([gE, gM, gH, 0.0])

    sizes = {int(c): int(n) for c, n in zip(*np.unique(assign, return_counts=True))}

    # prior toàn cục để shrink
    tot = np.zeros(4)
    for tl in probe_tiers.values():
        for t in tl:
            tot[TIERS.index(t)] += 1
    prior = tot / max(tot.sum(), 1.0)

    out: Dict[int, ClusterStat] = {}
    for cid, n in sizes.items():
        tl = probe_tiers.get(cid, [])
        counts = np.zeros(4)
        for t in tl:
            counts[TIERS.index(t)] += 1
        n0 = max(len(tl), 1)
        # Dirichlet posterior mean: (counts + m*prior) / (n0 + m)
        p = (counts + cfg.prior_strength * prior) / (n0 + cfg.prior_strength)
        se = float(np.sqrt(np.max(p * (1 - p)) / n0))

        dropped = p[TIERS.index(Tier.INVALID)] > cfg.invalid_drop
        value = float((p * gains).sum() * (1.0 + cfg.entropy_bonus * _entropy(p)))
        # trọng số = giá trị huấn luyện  x  nhiệt độ long-tail trên kích thước cụm
        w = 0.0 if dropped else value * (n ** alpha)
        out[cid] = ClusterStat(cid, n, len(tl), counts, p, value, w, dropped, se)
    return out


def expand_quota(stats: Dict[int, ClusterStat], budget: int,
                 cap_ratio: float = 0.6, floor: int = 8,
                 iters: int = 50) -> Dict[int, int]:
    """Water-filling: phân bổ `budget` theo weight, tôn trọng trần cap_ratio*N_c.

    Phần dư do cụm bị chạm trần được chia lại cho các cụm chưa chạm trần, lặp
    tới khi hội tụ — nên cụm nhỏ không bị lấy quá tay mà ngân sách vẫn dùng hết.
    """
    live = {c: s for c, s in stats.items() if not s.dropped and s.weight > 0}
    cap = {c: max(0, min(int(cap_ratio * s.size), s.size)) for c, s in live.items()}
    alloc = {c: 0 for c in live}
    rem = budget
    for _ in range(iters):
        free = [c for c in live if alloc[c] < cap[c]]
        W = sum(live[c].weight for c in free)
        if not free or W <= 0 or rem <= 0:
            break
        moved = 0
        for c in free:
            add = min(int(rem * live[c].weight / W), cap[c] - alloc[c])
            alloc[c] += add
            moved += add
        rem -= moved
        if moved == 0:
            break
    # sàn phủ: mỗi cụm sống ít nhất `floor` mẫu (nếu còn chỗ)
    for c, s in live.items():
        if alloc[c] < min(floor, cap[c]):
            alloc[c] = min(floor, cap[c])
    return alloc
