"""Orchestrator DDAS — nối Stage 1 (trang) và Stage 2 (element) thành một luồng.

Luồng chạy thật (mỗi bước là một job Spark/Ray riêng, checkpoint ra parquet):

  [0] ingest      PDF -> trang -> ảnh 200dpi + text layer (PyMuPDF, CPU)
  [1] embed       ViT-base + layout-prior  -> vector 512+24 chiều      (GPU, toàn pool)
  [2] cluster     K-Means phân cấp + làm phẳng mật độ + kênh tail      (GPU, fit trên mẫu)
  [3] dedup       LSH trong cụm, ngưỡng cosine                          (CPU)
  [4] probe       CMCV trên n0 trang/cụm -> p_c, loại cụm rác          (GPU, ~3% pool)
  [5] expand      quota theo cụm -> bốc tập ứng viên trang             (CPU)
  [6] cmcv_full   CMCV cascade trên tập ứng viên -> tier + nhãn tự động (GPU, chi phí chính)
  [7] final·layout lấy mẫu lồng nhau (cụm x độ khó) MỨC TRANG cho subtask layout (CPU) -- run()
  [8] layout_detect Docling Layout Heron trên candidate set -> bbox+class ĐỘC LẬP  (GPU, suy luận THẬT)
  [9] elements    derive_element_cmcv: tra nội dung [8] trong output [6] theo IoU  (CPU, 0 suy luận CMCV thêm) -- run_elements()
  [10] final·{text,formula,table} cluster + lấy mẫu lồng nhau MỨC ELEMENT (CPU + GPU nhẹ crop-embed)

[7] và [8]-[9] dùng chung candidate set từ [6] nhưng lấy mẫu ĐỘC LẬP theo quota
riêng mỗi subtask (SamplerConfig.budget) — một trang được chọn cho layout không
có nghĩa element của nó được chọn cho text/formula/table, và ngược lại.

  [11] judge_refine §3.3 — render-then-verify sửa nhãn Hard, phần không cứu được
       xếp ưu tiên sang chú thích tay (judge_refine.py) -- run_judge_refine()

Đầu ra: SFT set cho Easy/Medium (dùng ngay) + mẫu Hard đã được §3.3 sửa +
hàng đợi chú thích tay đã xếp ưu tiên, kèm toàn bộ metadata để tái lập.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, Hashable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from .cluster import ClusterIndex, dedup_within_cluster, hierarchical_cluster
from .cmcv import CMCVRecord, ParseResult, Tier
from .config import DDASConfig, SUBTASKS
from .element import Element, build_elements, cluster_and_sample, embed_elements
from .judge_refine import JudgeFn, JudgeRefine, prioritize, weakness_by_subtask
from .layout_heron import LayoutBox
from .probe import ClusterStat, cluster_weights, expand_quota, probe_size
from .sampler import Allocation, allocate_nested, draw
from .sft import HardItem, assemble_sft_set


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
        # cand_arr/tiers (candidate set ĐẦY ĐỦ, đã CMCV) trả kèm final (chỉ
        # tập con đã lọc theo quota layout) vì Stage 2 (run_elements) phải
        # dùng cand_arr — dùng nhầm final sẽ kế thừa bias lấy mẫu của layout.
        return final, idx, stats, alloc, cand_arr, tiers

    # ------------------------------------------------------- Stage 2 (element) --
    def run_elements(self, cand_pages: Sequence[str],
                     layout_fn: Callable[[str], Sequence[LayoutBox]],
                     parse_fn: Callable[[str], Tuple[ParseResult, ParseResult, Optional[ParseResult]]],
                     image_fn: Callable[[str], Image.Image],
                     page_wh_fn: Callable[[str], Tuple[float, float]],
                     crop_encoder,
                     budget: Dict[str, int] | None = None,
                     seed: int = 0) -> Dict[str, List[Element]]:
        """Stage 2 — text/formula/table lấy mẫu Ở MỨC ELEMENT, tách khỏi quota
        trang của run() (dùng cho layout). `cand_pages` PHẢI là `cand_arr`
        (giá trị thứ 5 `run()` trả về — candidate set ĐẦY ĐỦ, đã CMCV), ép
        sang str để làm page_id. KHÔNG dùng `final` (giá trị thứ 1) — đó là
        tập đã lọc theo quota riêng của layout, dùng nhầm sẽ kế thừa bias.

        Đúng Figure 3 paper: `layout_fn` (HeronLayoutDetector) chạy TRƯỚC,
        cho bbox+class độc lập với 3 model CMCV — đây là suy luận THẬT, có
        chi phí (xem costmodel.py). `parse_fn` PHẢI trả lại ParseResult đã
        cache từ CMCV trang (không gọi model/API lần 2).
        """
        elements = build_elements(list(cand_pages), layout_fn, parse_fn, self.cfg.cmcv)
        self.reports.append(StageReport("elements", len(cand_pages), len(elements),
                                        {"element/trang": len(elements) / max(len(cand_pages), 1)}))

        embeds = embed_elements(elements, image_fn, crop_encoder)
        page_wh = {pid: page_wh_fn(pid) for pid in set(str(p) for p in cand_pages)}

        budget = budget or self.cfg.sampler.budget
        out: Dict[str, List[Element]] = {}
        for st in ("text", "formula", "table"):
            chosen, ci, alloc = cluster_and_sample(
                elements, embeds, page_wh, st, self.cfg.cluster, self.cfg.sampler,
                self.cfg.probe.gain, budget[st], seed=seed)
            out[st] = [elements[i] for i in chosen]
            n_avail = sum(1 for e in elements if e.etype == st)
            extra = {} if alloc is None else {"phủ cụm": alloc.coverage,
                                              **{t.value: v for t, v in alloc.tier_mix.items()}}
            self.reports.append(StageReport(f"final·{st}", n_avail, len(chosen), extra))
        return out

    # --------------------------------------------------------- Final sampling --
    def assemble_sft_set(self, final_layout_page_ids: Sequence[int],
                         tiers_by_page: Dict[int, Tier],
                         parse_fn: Callable[[str], Tuple[ParseResult, ParseResult, Optional[ParseResult]]],
                         elements_by_subtask: Dict[str, List[Element]]):
        """Gộp layout (run()) + text/formula/table (run_elements()) thành 1 bộ
        SFT set duy nhất, tra pseudo-label đúng theo tier. Xem ddas.sft.

        `final_layout_page_ids`/`tiers_by_page` = giá trị thứ 1 và (`cand_arr`
        zip `tiers`, giá trị thứ 5+6) mà `run()` trả về.
        """
        ready, hard = assemble_sft_set(final_layout_page_ids, tiers_by_page,
                                       parse_fn, elements_by_subtask)
        self.reports.append(StageReport("sft_set", len(ready) + len(hard), len(ready),
                                        {"hard_queue": len(hard)}))
        return ready, hard

    # ------------------------------------------- §3.3 Judge-and-Refine --------
    def run_judge_refine(self, hard_queue: Sequence[HardItem],
                         judge_fn: JudgeFn,
                         image_fn: Callable[[str], Image.Image],
                         cmcv_records: Optional[Sequence[CMCVRecord]] = None,
                         expert_budget: Optional[int] = None):
        """Tiêu thụ hàng đợi Hard của assemble_sft_set() — xem ddas.judge_refine.

        Trả về (refined, expert): `refined` là mẫu Hard đã sửa được tự động,
        `.to_sft()` để gộp vào SFT set; `expert` là hàng đợi chú thích tay ĐÃ
        xếp ưu tiên, cắt theo `expert_budget`.

        `cmcv_records` (từ §3.2) chỉ dùng để tính độ yếu theo subtask cho tiêu
        chí ưu tiên #2 — bỏ trống thì chỉ xếp theo tiêu chí #1.
        """
        cfg = self.cfg.judge
        jr = JudgeRefine(judge_fn, image_fn, cfg)
        refined, expert = jr.run(hard_queue)
        weakness = weakness_by_subtask(cmcv_records) if cmcv_records else None
        expert = prioritize(expert, weakness, cfg.min_confidence,
                            expert_budget or cfg.expert_budget)
        self.reports.append(StageReport("judge_refine", len(hard_queue), len(refined),
                                        {"người": len(expert),
                                         "tự cứu %": 100 * jr.resolve_rate,
                                         "vòng/mẫu": jr.mean_rounds,
                                         "render lỗi": jr.stats["render_failed"]}))
        return refined, expert

    def summary(self) -> str:
        return "\n".join(str(r) for r in self.reports)
