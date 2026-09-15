"""Orchestrator DDAS — nối Stage 1 (trang) và Stage 2 (element) thành một luồng.

Luồng chạy thật (mỗi bước là một job Spark/Ray riêng, checkpoint ra parquet):

  [0] ingest      PDF -> trang -> ảnh 200dpi + text layer (PyMuPDF, CPU)
  [1] embed       ViT-base + layout-prior  -> vector 512+24 chiều      (GPU, toàn pool)
  [2] cluster     K-Means phân cấp + làm phẳng mật độ + kênh tail      (GPU, fit trên mẫu)
  [3] dedup       LSH trong cụm, ngưỡng cosine                          (CPU)
  [4] probe       CMCV trên n0 trang/cụm -> p_c, loại cụm rác          (GPU, ~3% pool)
  [5] expand      quota theo cụm -> bốc tập ứng viên trang             (CPU)
  [6] cmcv_full   CMCV cascade trên tập ứng viên -> tier + nhãn tự động (GPU, chi phí chính)
  [7] elements    dẫn xuất element-CMCV từ [6] + nhúng crop            (CPU + GPU nhẹ)
  [8] final       lấy mẫu lồng nhau (cụm x độ khó) cho 4 subtask       (CPU)

Đầu ra: SFT set cho Easy/Medium (dùng ngay) + hàng đợi Hard chuyển sang Phần 3
(Judge-and-Refine), kèm toàn bộ metadata để tái lập.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, Hashable, List, Sequence

import numpy as np

from .cluster import ClusterIndex, dedup_within_cluster, hierarchical_cluster
from .cmcv import Tier
from .config import DDASConfig, SUBTASKS
from .probe import ClusterStat, cluster_weights, expand_quota, probe_size
from .sampler import Allocation, allocate_nested, draw


@dataclass
class StageReport:
    name: str
    n_in: int
    n_out: int
    extra: Dict[str, float] = field(default_factory=dict)

    def __str__(self) -> str:
        e = "  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                      for k, v in self.extra.items())
        return f"[{self.name:<10}] {self.n_in:>12,} -> {self.n_out:>12,}   {e}"


class DDASPipeline:
    """Điều phối. Các lời gọi nặng (embed, CMCV) được inject để test/chạy thật dùng chung code."""

    def __init__(self, cfg: DDASConfig,
                 cmcv_fn: Callable[[Sequence[int]], List[Tier]],
                 subtask: str = "text"):
        self.cfg = cfg
        self.cmcv_fn = cmcv_fn          # batch page ids -> tier (thực tế: CMCV cascade trên GPU)
        self.subtask = subtask
        self.reports: List[StageReport] = []

    # ---------------------------------------------------------------- run --
    def run(self, X: np.ndarray, budget: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        N = len(X)

        idx: ClusterIndex = hierarchical_cluster(X, self.cfg.cluster, seed=seed)
        self.reports.append(StageReport("cluster", N, len(idx.sizes),
                                        {"cụm": len(idx.sizes),
                                         "tail": idx.sizes.get(-1, 0)}))

        keep = dedup_within_cluster(X, idx.assign, self.cfg.cluster.dedup_cosine, seed)
        self.reports.append(StageReport("dedup", N, int(keep.sum()),
                                        {"loại bỏ %": 100 * (1 - keep.mean())}))

        # --- probe: chỉ CMCV trên mẫu dò ---------------------------------
        probe_tiers: Dict[int, List[Tier]] = {}
        n_probe = 0
        for cid in idx.sizes:
            m = np.where((idx.assign == cid) & keep)[0]
            if len(m) == 0:
                continue
            n0 = probe_size(len(m), self.cfg.probe)
            sel = rng.choice(m, n0, replace=False)
            probe_tiers[cid] = self.cmcv_fn(sel)
            n_probe += n0
        stats = cluster_weights(idx.assign, probe_tiers, self.subtask,
                                self.cfg.probe, self.cfg.sampler.alpha)
        dropped = {c for c, s in stats.items() if s.dropped}
        self.reports.append(StageReport("probe", N, n_probe,
                                        {"% pool": 100 * n_probe / N,
                                         "cụm loại": len(dropped),
                                         "SE tb": float(np.mean([s.se for s in stats.values()]))}))

        # --- expand: bốc tập ứng viên trang theo quota cụm ----------------
        quota = expand_quota(stats, self.cfg.page_budget if budget is None else budget * 3,
                             self.cfg.sampler.cap_ratio, self.cfg.sampler.floor_per_cell)
        cand: List[int] = []
        for cid, q in quota.items():
            m = np.where((idx.assign == cid) & keep)[0]
            if q > 0 and len(m):
                cand.extend(rng.choice(m, min(q, len(m)), replace=False).tolist())
        cand_arr = np.array(cand, dtype=int)
        ratio = len(cand_arr) / max(budget, 1)
        if ratio < 2.0:
            # Chế độ hỏng thật: khi tập ứng viên xấp xỉ ngân sách, bộ lấy mẫu cuối
            # không còn gì để chọn -> quota chạm trần ở mọi ô và trộn độ khó suy biến
            # về đúng phân bố của pool. Cần >=3x để chiều độ khó có tác dụng.
            print(f"  [CẢNH BÁO] ứng viên/ngân sách = {ratio:.1f}x (<2x): "
                  f"tăng expand hoặc giảm budget, nếu không việc phân tầng độ khó sẽ vô hiệu")
        self.reports.append(StageReport("expand", N, len(cand_arr),
                                        {"ứng viên/ngân sách": ratio}))

        # --- CMCV đầy đủ trên tập ứng viên: vừa ra tier vừa ra nhãn -------
        tiers = self.cmcv_fn(cand_arr)
        avail: Dict = defaultdict(int)
        members: Dict = defaultdict(list)
        for pid, t in zip(cand_arr, tiers):
            cell = (int(idx.assign[pid]), t)
            avail[cell] += 1
            members[cell].append(int(pid))
        mix = {t: sum(v for (_k, d), v in avail.items() if d == t) / max(len(cand_arr), 1)
               for t in (Tier.EASY, Tier.MEDIUM, Tier.HARD)}
        self.reports.append(StageReport("cmcv_full", len(cand_arr), len(cand_arr),
                                        {t.value: v for t, v in mix.items()}))

        # --- lấy mẫu cuối: lồng nhau (cụm x độ khó) ----------------------
        alloc: Allocation = allocate_nested(dict(avail), self.subtask,
                                            self.cfg.sampler, self.cfg.probe.gain,
                                            idx.paths, budget=budget)
        final = draw(alloc, members, seed=seed + 1)
        self.reports.append(StageReport("final", len(cand_arr), len(final),
                                        {"phủ cụm": alloc.coverage,
                                         **{t.value: v for t, v in alloc.tier_mix.items()}}))
        return final, idx, stats, alloc

    def summary(self) -> str:
        return "\n".join(str(r) for r in self.reports)
