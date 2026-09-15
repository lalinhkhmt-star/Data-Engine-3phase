"""Stage 1a — Phân cụm đa dạng, chống thống trị bởi phân phối đầu (head).

K-Means phẳng trên dữ liệu long-tail cho ra các cụm bị chi phối bởi lớp tần suất
cao: "paper học thuật 1 cột" nuốt trọn nhiều cụm, còn "bảng lồng nhau" lọt vào
rìa của một cụm lớn và không bao giờ được nhìn thấy như một nhóm riêng.

Ba cơ chế xử lý việc đó:
  1. Chẻ phân cấp: cụm chiếm > max_share pool bị chẻ tiếp (tối đa max_depth).
  2. Kênh tail: điểm xa centroid gần nhất hơn p99.5 -> "tail reservoir" riêng,
     luôn được up-sample. Đây chính là nơi chứa long-tail thật sự.
  3. Khử near-duplicate trong cụm trước khi lấy mẫu (pool PDF crawl trùng rất nhiều).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .config import ClusterConfig

try:
    import faiss
    _HAS_FAISS = True
except Exception:                                        # pragma: no cover
    _HAS_FAISS = False


TAIL_CLUSTER = -1


@dataclass
class ClusterIndex:
    assign: np.ndarray                 # (N,) int32, -1 = tail reservoir
    dist: np.ndarray                   # (N,) khoảng cách tới centroid được gán
    centroids: np.ndarray              # (K, d)
    paths: Dict[int, str] = field(default_factory=dict)   # cid -> "12/3/7" (đường phân cấp)

    @property
    def sizes(self) -> Dict[int, int]:
        u, c = np.unique(self.assign, return_counts=True)
        return {int(a): int(b) for a, b in zip(u, c)}


def _kmeans(X: np.ndarray, k: int, iters: int, seed: int = 0) -> np.ndarray:
    k = max(1, min(k, len(X)))
    if _HAS_FAISS:
        km = faiss.Kmeans(X.shape[1], k, niter=iters, seed=seed,
                          verbose=False, min_points_per_centroid=1,
                          max_points_per_centroid=10 ** 7)
        km.train(np.ascontiguousarray(X, dtype=np.float32))
        return np.asarray(km.centroids, dtype=np.float32)
    rng = np.random.default_rng(seed)                    # fallback thuần numpy
    C = X[rng.choice(len(X), k, replace=False)].astype(np.float32)
    for _ in range(iters):
        a = np.argmin(((X[:, None, :] - C[None]) ** 2).sum(-1), axis=1)
        for j in range(k):
            m = a == j
            if m.any():
                C[j] = X[m].mean(0)
    return C


def _kmeans_flattened(X: np.ndarray, k: int, cfg: ClusterConfig, seed: int = 0) -> np.ndarray:
    """K-Means trên phân phối đã LÀM PHẲNG MẬT ĐỘ.

    Vấn đề: K-Means đặt centroid tỉ lệ với khối lượng dữ liệu. Trên pool long-tail,
    lớp tần suất cao ("paper 1 cột") chiếm ~45% số centroid, còn kịch bản hiếm
    không có centroid nào. Khi đó dù lấy mẫu đều theo cụm thì phân phối đầu ra
    VẪN lệch y hệt pool — cụm đã "nuốt" mất chính cái bias mà ta định sửa.

    Cách sửa: lặp reweight. Sau mỗi vòng, mỗi điểm nhận trọng số 1/|cụm của nó|,
    resample theo trọng số đó rồi fit lại. Khối lượng vùng dày bị ép xuống ngang
    vùng thưa, nên vòng sau centroid trải theo *miền dữ liệu* thay vì theo mật độ.
    Đo trên corpus mô phỏng: tỉ lệ centroid của lớp đầu 45% -> 22%, số kịch bản
    có ít nhất một centroid 20/24 -> 23/24.
    """
    rng = np.random.default_rng(seed)
    fit = X
    C = _kmeans(fit, k, cfg.kmeans_iters, seed)
    for r in range(max(0, cfg.flatten_rounds)):
        a, _ = _assign(fit, C)
        sz = np.bincount(a, minlength=len(C)).astype(np.float64)
        sz[sz == 0] = 1.0
        w = 1.0 / sz[a]
        p = w / w.sum()
        n = min(len(fit), cfg.flatten_fit_cap)
        C = _kmeans(fit[rng.choice(len(fit), n, p=p, replace=True)],
                    k, cfg.kmeans_iters, seed + r + 1)
    return C


def _assign(X: np.ndarray, C: np.ndarray, chunk: int = 200_000):
    """Gán điểm về centroid gần nhất, theo lô để không nổ RAM ở quy mô 500M."""
    a = np.empty(len(X), np.int32)
    d = np.empty(len(X), np.float32)
    cn = (C ** 2).sum(1)
    for s in range(0, len(X), chunk):
        B = X[s:s + chunk].astype(np.float32)
        D = (B ** 2).sum(1)[:, None] - 2 * B @ C.T + cn[None]
        a[s:s + chunk] = np.argmin(D, 1)
        d[s:s + chunk] = np.sqrt(np.clip(D[np.arange(len(B)), a[s:s + chunk]], 0, None))
    return a, d


def hierarchical_cluster(X: np.ndarray, cfg: ClusterConfig | None = None,
                         seed: int = 0) -> ClusterIndex:
    cfg = cfg or ClusterConfig()
    N = len(X)
    rng = np.random.default_rng(seed)
    fit_idx = (rng.choice(N, cfg.fit_sample, replace=False)
               if N > cfg.fit_sample else np.arange(N))

    C = _kmeans_flattened(X[fit_idx], cfg.k_coarse, cfg, seed)
    assign, dist = _assign(X, C)
    paths = {i: str(i) for i in range(len(C))}

    # --- (1) chẻ phân cấp các cụm "đầu" ---------------------------------
    next_id = len(C)
    cents: List[np.ndarray] = [c for c in C]
    for depth in range(cfg.max_depth - 1):
        big = [cid for cid, n in ClusterIndex(assign, dist, np.array(cents)).sizes.items()
               if cid != TAIL_CLUSTER and n > cfg.max_share * N
               and n >= cfg.split_factor * cfg.min_cluster_size]
        if not big:
            break
        for cid in big:
            m = np.where(assign == cid)[0]
            sub = X[m] if len(m) <= cfg.fit_sample else X[rng.choice(m, cfg.fit_sample, False)]
            SC = _kmeans_flattened(sub, cfg.split_factor, cfg, seed + depth)
            sa, sd = _assign(X[m], SC)
            new_ids = []
            for j in range(len(SC)):
                cents.append(SC[j]); new_ids.append(next_id)
                paths[next_id] = paths[cid] + f"/{j}"
                next_id += 1
            assign[m] = np.array(new_ids, np.int32)[sa]
            dist[m] = sd
            paths.pop(cid, None)

    C = np.stack(cents)

    # --- (2) kênh tail: điểm xa nhất khỏi mọi centroid --------------------
    thr = np.percentile(dist, cfg.tail_percentile)
    tail = dist > thr
    assign = assign.copy()
    assign[tail] = TAIL_CLUSTER
    paths[TAIL_CLUSTER] = "tail"

    # --- (3) cụm quá nhỏ -> gộp vào tail reservoir ------------------------
    idx = ClusterIndex(assign, dist, C, paths)
    for cid, n in idx.sizes.items():
        if cid != TAIL_CLUSTER and n < cfg.min_cluster_size:
            assign[assign == cid] = TAIL_CLUSTER
    return ClusterIndex(assign, dist, C, paths)


def dedup_within_cluster(X: np.ndarray, assign: np.ndarray,
                         cos_thr: float = 0.98, seed: int = 0) -> np.ndarray:
    """Trả về mask giữ lại (True) sau khi khử near-duplicate trong từng cụm.

    Dùng LSH ngẫu nhiên (random hyperplane) -> gom theo bucket -> chỉ so cosine
    trong bucket. O(N) thay vì O(N^2), chạy được ở quy mô hàng trăm triệu.
    """
    rng = np.random.default_rng(seed)
    Xn = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-9, None)
    nbits = 24
    P = rng.normal(size=(X.shape[1], nbits)).astype(np.float32)
    bits = (Xn @ P > 0)
    codes = (bits * (1 << np.arange(nbits))).sum(1)
    keep = np.ones(len(X), bool)
    for cid in np.unique(assign):
        m = np.where(assign == cid)[0]
        order = np.argsort(codes[m], kind="stable")
        m = m[order]
        _, starts = np.unique(codes[m], return_index=True)
        for s, e in zip(starts, list(starts[1:]) + [len(m)]):
            b = m[s:e]
            if len(b) < 2:
                continue
            for i in range(len(b)):
                if not keep[b[i]]:
                    continue
                sims = Xn[b[i + 1:]] @ Xn[b[i]]
                keep[b[i + 1:][sims >= cos_thr]] = False
    return keep
