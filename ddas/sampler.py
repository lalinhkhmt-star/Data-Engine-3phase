"""Final sampling — cân bằng đồng thời trên không gian (subtask x cluster x difficulty).

Bài toán: cho các ô (cell) c = (subtask t, cluster k, tier d) với N_tkd mẫu sẵn có,
chọn quota q_tkd sao cho
    max  sum_c  q_c * log-utility        s.t.  sum_k,d q_tkd = B_t,  q_c <= cap*N_c
với utility = gain độ khó g_d^(t)  x  hiệu chỉnh long-tail  N_k^(alpha-1).

Nghiệm dạng đóng của bài toán này là  q_c ∝ g_d * N_k^alpha  (alpha<1 chính là
"nhiệt độ" ép phẳng long-tail: alpha=1 giữ nguyên phân phối gốc, alpha=0 chia đều).
Ràng buộc trần/sàn xử lý bằng water-filling nên ngân sách luôn được tiêu hết.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Hashable, List, Tuple

import numpy as np

from .cmcv import Tier
from .config import SamplerConfig

Cell = Tuple[int, Tier]                       # (cluster_id, tier)


@dataclass
class Allocation:
    quota: Dict[Cell, int]
    available: Dict[Cell, int]
    budget: int
    tier_mix: Dict[Tier, float]
    coverage: float                           # tỉ lệ cluster được phủ >0 mẫu
    entropy_gain: float                       # H(quota)/H(uniform) so với H(pool)


def _entropy(counts: np.ndarray) -> float:
    p = counts / max(counts.sum(), 1e-9)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def allocate(available: Dict[Cell, int], subtask: str,
             cfg: SamplerConfig, gains: Dict[str, Tuple[float, float, float]],
             budget: int | None = None) -> Allocation:
    budget = budget if budget is not None else cfg.budget[subtask]
    gE, gM, gH = gains[subtask]
    gmap = {Tier.EASY: gE, Tier.MEDIUM: gM, Tier.HARD: gH, Tier.INVALID: 0.0}

    cluster_size: Dict[int, int] = defaultdict(int)
    for (k, _d), n in available.items():
        cluster_size[k] += n

    w = {c: gmap[c[1]] * (cluster_size[c[0]] ** cfg.alpha)
         for c in available if gmap[c[1]] > 0 and available[c] > 0}
    cap = {c: max(0, int(cfg.cap_ratio * available[c])) for c in w}

    alloc = {c: 0 for c in w}
    rem = budget
    for _ in range(cfg.waterfill_iters):
        free = [c for c in w if alloc[c] < cap[c]]
        W = sum(w[c] for c in free)
        if not free or W <= 0 or rem <= 0:
            break
        moved = 0
        for c in free:
            add = min(int(rem * w[c] / W), cap[c] - alloc[c])
            alloc[c] += add
            moved += add
        if moved == 0:                         # phần dư nhỏ: rải round-robin
            for c in sorted(free, key=lambda c: -w[c]):
                if rem <= 0:
                    break
                alloc[c] += 1; rem -= 1
            break
        rem -= moved

    # sàn phủ: đảm bảo mọi cluster hợp lệ xuất hiện trong tập huấn luyện
    for c in w:
        if alloc[c] < min(cfg.floor_per_cell, cap[c]):
            alloc[c] = min(cfg.floor_per_cell, cap[c])

    tot = max(sum(alloc.values()), 1)
    mix = defaultdict(float)
    for c, q in alloc.items():
        mix[c[1]] += q / tot
    covered = len({c[0] for c, q in alloc.items() if q > 0})
    n_clusters = len(cluster_size)

    pool_by_cluster = np.array([cluster_size[k] for k in cluster_size], float)
    out_by_cluster = np.zeros(n_clusters)
    keys = list(cluster_size)
    for c, q in alloc.items():
        out_by_cluster[keys.index(c[0])] += q
    hmax = np.log(n_clusters)
    gain = (_entropy(out_by_cluster) - _entropy(pool_by_cluster)) / max(hmax, 1e-9)

    return Allocation(alloc, dict(available), budget, dict(mix),
                      covered / max(n_clusters, 1), gain)


def draw(alloc: Allocation, members: Dict[Cell, List[Hashable]],
         seed: int = 0) -> List[Hashable]:
    """Bốc mẫu thật theo quota. Cell thiếu mẫu -> lấy hết (không lặp lại)."""
    rng = np.random.default_rng(seed)
    out: List[Hashable] = []
    for c, q in alloc.quota.items():
        pool = members.get(c, [])
        if q <= 0 or not pool:
            continue
        take = min(q, len(pool))
        out.extend(np.asarray(pool, dtype=object)[rng.choice(len(pool), take, replace=False)].tolist())
    return out


# ------------------------------------------------- phân bổ theo cây ---------

def hierarchical_weights(paths: Dict[int, str], sizes: Dict[int, int],
                         alpha: float = 0.4) -> Dict[int, float]:
    """Trọng số đa dạng tính TRÊN CÂY phân cụm, không phải trên danh sách cụm phẳng.

    Vì sao cần: chẻ phân cấp khiến một kịch bản "đầu" (vd. paper 1 cột) bị tách
    thành hàng chục cụm con. Nếu chia ngân sách đều theo *cụm lá*, phần đầu vẫn
    nhận gấp hàng chục lần phần đuôi — tức là đã chẻ cụm nhưng long-tail shift
    KHÔNG hề được sửa. Đây là cái bẫy dễ bỏ sót nhất của cluster sampling.

    Cách làm: rót khối lượng từ gốc xuống. Ở mỗi nút, chia cho các nhánh con
    theo (kích thước cây con)^alpha. Trọng số lá = tích dọc đường đi.
    alpha=1 -> đúng tỉ lệ pool; alpha=0 -> chia đều theo nhánh ở MỌI mức.
    """
    tree: Dict[str, List[str]] = defaultdict(list)
    sub: Dict[str, float] = defaultdict(float)
    leaves: Dict[str, int] = {}
    for cid, path in paths.items():
        if cid not in sizes:
            continue
        leaves[path] = cid
        parts = path.split("/")
        for i in range(len(parts)):
            node = "/".join(parts[:i + 1])
            sub[node] += sizes[cid]
            parent = "/".join(parts[:i]) if i else ""
            if node not in tree[parent]:
                tree[parent].append(node)
    sub[""] = sum(sizes.get(c, 0) for c in paths if c in sizes)

    w: Dict[int, float] = {}

    def pour(node: str, mass: float) -> None:
        if node in leaves:
            w[leaves[node]] = mass
            return
        kids = tree.get(node, [])
        if not kids:
            return
        z = sum(sub[k] ** alpha for k in kids) or 1.0
        for k in kids:
            pour(k, mass * (sub[k] ** alpha) / z)

    pour("", 1.0)
    return w


def allocate_tree(available: Dict[Cell, int], subtask: str, cfg: SamplerConfig,
                  gains: Dict[str, Tuple[float, float, float]],
                  paths: Dict[int, str], budget: int | None = None) -> Allocation:
    """Như allocate() nhưng chiều diversity dùng trọng số cây (hierarchical_weights)."""
    budget = budget if budget is not None else cfg.budget[subtask]
    gE, gM, gH = gains[subtask]
    gmap = {Tier.EASY: gE, Tier.MEDIUM: gM, Tier.HARD: gH, Tier.INVALID: 0.0}

    cluster_size: Dict[int, int] = defaultdict(int)
    for (k, _d), n in available.items():
        cluster_size[k] += n
    hw = hierarchical_weights(paths, dict(cluster_size), cfg.alpha)

    w = {c: gmap[c[1]] * hw.get(c[0], 0.0)
         for c in available if gmap[c[1]] > 0 and available[c] > 0}
    w = {c: v for c, v in w.items() if v > 0}
    cap = {c: max(0, int(cfg.cap_ratio * available[c])) for c in w}

    alloc = {c: 0 for c in w}
    rem = budget
    for _ in range(cfg.waterfill_iters):
        free = [c for c in w if alloc[c] < cap[c]]
        W = sum(w[c] for c in free)
        if not free or W <= 0 or rem <= 0:
            break
        moved = 0
        for c in free:
            add = min(int(rem * w[c] / W), cap[c] - alloc[c])
            alloc[c] += add; moved += add
        if moved == 0:
            for c in sorted(free, key=lambda c: -w[c]):
                if rem <= 0:
                    break
                alloc[c] += 1; rem -= 1
            break
        rem -= moved
    for c in w:
        if alloc[c] < min(cfg.floor_per_cell, cap[c]):
            alloc[c] = min(cfg.floor_per_cell, cap[c])

    tot = max(sum(alloc.values()), 1)
    mix: Dict[Tier, float] = defaultdict(float)
    for c, q in alloc.items():
        mix[c[1]] += q / tot
    covered = len({c[0] for c, q in alloc.items() if q > 0})
    keys = list(cluster_size)
    pool_by = np.array([cluster_size[k] for k in keys], float)
    out_by = np.zeros(len(keys))
    for c, q in alloc.items():
        out_by[keys.index(c[0])] += q
    gain = (_entropy(out_by) - _entropy(pool_by)) / max(np.log(len(keys)), 1e-9)
    return Allocation(alloc, dict(available), budget, dict(mix),
                      covered / max(len(cluster_size), 1), gain)


def allocate_nested(available: Dict[Cell, int], subtask: str, cfg: SamplerConfig,
                    gains: Dict[str, Tuple[float, float, float]],
                    paths: Dict[int, str], budget: int | None = None,
                    outer_iters: int = 12,
                    cluster_prior: Dict[int, float] | None = None) -> Allocation:
    """Phân bổ LỒNG NHAU — cách đúng để "cân bằng đồng thời" hai chiều.

    Vì sao không dùng một lần water-fill với w = gain_d * w_cluster:
    hai chiều khi đó cạnh tranh trên cùng một ngân sách, và chiều difficulty
    (tỉ lệ gain tới 10:1) áp đảo chiều diversity (tỉ lệ ~2-3:1). Hệ quả đo được:
    cụm ĐẦU lại hút thêm quota, vì nó chứa nhiều mẫu Medium/Hard nhất về số tuyệt
    đối — long-tail shift bị làm tệ đi đúng lúc ta tưởng đang sửa nó.

    Thay vào đó:
      Bước ngoài  — chia ngân sách cho các CỤM theo trọng số cây (chỉ diversity).
      Bước trong  — trong mỗi cụm, chia quota của cụm đó theo gain độ khó.
    Hai chiều trực giao: trộn độ khó được áp dụng *có điều kiện theo cụm*, nên
    không cụm nào có thể mua thêm quota bằng cách "giàu mẫu Hard".
    Quota mà một cụm không hấp thụ hết được trả lại và chia vòng ngoài kế tiếp.
    """
    budget = budget if budget is not None else cfg.budget[subtask]
    gE, gM, gH = gains[subtask]
    gmap = {Tier.EASY: gE, Tier.MEDIUM: gM, Tier.HARD: gH, Tier.INVALID: 0.0}

    by_cluster: Dict[int, Dict[Tier, int]] = defaultdict(dict)
    for (k, d), n in available.items():
        if n > 0 and gmap[d] > 0:
            by_cluster[k][d] = n
    cluster_size = {k: sum(v.values()) for k, v in by_cluster.items()}
    hw = hierarchical_weights(paths, cluster_size, cfg.alpha)
    if cluster_prior:                       # hệ số hiếm cục bộ (ddas.density)
        hw = {k: v * cluster_prior.get(k, 1.0) for k, v in hw.items()}
        z = sum(hw.values()) or 1.0
        hw = {k: v / z for k, v in hw.items()}
    hw = {k: v for k, v in hw.items() if k in by_cluster and v > 0}

    cell_cap = {(k, d): max(0, int(cfg.cap_ratio * n))
                for k, v in by_cluster.items() for d, n in v.items()}
    cluster_cap = {k: sum(cell_cap[(k, d)] for d in by_cluster[k]) for k in by_cluster}

    alloc: Dict[Cell, int] = {c: 0 for c in cell_cap}
    rem = budget
    for _ in range(outer_iters):
        if rem <= 0:
            break
        free_k = [k for k in hw if sum(alloc[(k, d)] for d in by_cluster[k]) < cluster_cap[k]]
        W = sum(hw[k] for k in free_k)
        if not free_k or W <= 0:
            break
        moved = 0
        for k in free_k:
            used_k = sum(alloc[(k, d)] for d in by_cluster[k])
            Qk = min(int(rem * hw[k] / W), cluster_cap[k] - used_k)
            if Qk <= 0:
                continue
            # --- bước trong: chia Qk theo gain độ khó, water-fill trong cụm ---
            tiers = list(by_cluster[k])
            inner = Qk
            for _ in range(len(tiers) + 2):
                free_d = [d for d in tiers if alloc[(k, d)] < cell_cap[(k, d)]]
                Wd = sum(gmap[d] for d in free_d)
                if not free_d or Wd <= 0 or inner <= 0:
                    break
                step = 0
                for d in free_d:
                    add = min(int(inner * gmap[d] / Wd), cell_cap[(k, d)] - alloc[(k, d)])
                    alloc[(k, d)] += add; step += add
                if step == 0:
                    for d in sorted(free_d, key=lambda d: -gmap[d]):
                        if inner <= 0:
                            break
                        alloc[(k, d)] += 1; inner -= 1; step += 1
                    break
                inner -= step
            moved += Qk - inner
        if moved == 0:
            break
        rem -= moved

    for (k, d), c in cell_cap.items():
        if alloc[(k, d)] < min(cfg.floor_per_cell, c):
            alloc[(k, d)] = min(cfg.floor_per_cell, c)

    tot = max(sum(alloc.values()), 1)
    mix: Dict[Tier, float] = defaultdict(float)
    for (k, d), q in alloc.items():
        mix[d] += q / tot
    covered = len({k for (k, _d), q in alloc.items() if q > 0})
    keys = list(cluster_size)
    pool_by = np.array([cluster_size[k] for k in keys], float)
    out_by = np.zeros(len(keys))
    for (k, _d), q in alloc.items():
        out_by[keys.index(k)] += q
    gain = (_entropy(out_by) - _entropy(pool_by)) / max(np.log(len(keys)), 1e-9)
    return Allocation(alloc, dict(available), budget, dict(mix),
                      covered / max(len(cluster_size), 1), gain)
