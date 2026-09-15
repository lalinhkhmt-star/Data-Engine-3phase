"""Benchmark DDAS trên corpus tài liệu tổng hợp có ground-truth ẩn.

So sánh ở CÙNG ngân sách:
  A. Uniform          — lấy ngẫu nhiên (baseline)
  B. Cluster-only     — chia đều theo cụm (chỉ diversity, kiểu cluster sampling cũ)
  C. DDAS (cụm+khó)   — phân bổ lồng nhau trên cây cụm x độ khó
  D. DDAS + rarity    — thêm hệ số thưa cục bộ vào trọng số cụm

Chạy trên HAI chế độ corpus để kiểm tra độ nhạy của giả định:
  R1 "thực tế"  — lớp phổ biến đồng nhất (dày), lớp hiếm dị biệt (thưa)
  R2 "trung lập"— độ trải rộng không tương quan với tần suất
"""
import sys, time
from collections import defaultdict

import numpy as np

sys.path.insert(0, "/home/linhlt109/Documents/Data_Engine")
from ddas.cluster import hierarchical_cluster, TAIL_CLUSTER
from ddas.cmcv import Tier
from ddas.config import ClusterConfig, ProbeConfig, SamplerConfig
from ddas.density import cluster_rarity, local_sparsity, rarity_regime_check
from ddas.probe import cluster_weights, probe_size
from ddas.sampler import allocate_nested, draw

N_SCEN, N_POOL, DIM, BUDGET = 24, 300_000, 48, 30_000
TIERS = [Tier.EASY, Tier.MEDIUM, Tier.HARD, Tier.INVALID]
GAIN = ProbeConfig().gain["text"]
G = np.array([*GAIN, 0.0])


def make_corpus(regime: str, seed: int = 7):
    rng = np.random.default_rng(seed)
    zipf = 1.0 / np.arange(1, N_SCEN + 1) ** 1.25
    zipf /= zipf.sum()
    centers = rng.normal(0, 5.0, (N_SCEN, DIM))
    if regime == "R1":
        # thực tế: kịch bản càng hiếm càng dị biệt -> trải rộng hơn
        spread = 0.35 + 0.95 * (np.arange(N_SCEN) / (N_SCEN - 1))
    else:
        spread = rng.uniform(0.4, 1.1, N_SCEN)
    dp = np.zeros((N_SCEN, 4))
    for s in range(N_SCEN):
        r = s / (N_SCEN - 1)
        e = np.clip(0.80 - 0.65 * r, .05, .95)
        m = np.clip(0.13 + 0.18 * r, .02, .5)
        h = np.clip(0.05 + 0.42 * r, .01, .6)
        dp[s] = np.array([e, m, h, .02]) / (e + m + h + .02)
    for s in (5, 17):                                   # kịch bản rác
        dp[s] = np.array([.05, .02, .03, .90])
    scen = rng.choice(N_SCEN, N_POOL, p=zipf)
    X = (centers[scen] + rng.normal(0, 1, (N_POOL, DIM)) * spread[scen][:, None]).astype(np.float32)
    tier = np.array([rng.choice(4, p=dp[s]) for s in scen])
    return X, scen, tier, zipf


def evaluate(sel, scen, tier):
    cnt = np.bincount(scen[sel], minlength=N_SCEN)
    p = cnt / cnt.sum(); q = p[p > 0]
    mix = [100 * (tier[sel] == t).mean() for t in range(4)]
    return dict(eff=float(np.exp(-(q * np.log(q)).sum())),
                head=100 * cnt[0] / cnt.sum(),
                tail=100 * cnt[-10:].sum() / cnt.sum(),
                cover=int((cnt > 0).sum()), waste=mix[3], mix=mix,
                signal=float(G[tier[sel]].mean()))


def run_regime(regime: str):
    X, scen, tier, zipf = make_corpus(regime)
    rng = np.random.default_rng(3)
    print(f"\n{'='*106}\nCHẾ ĐỘ {regime} — " +
          ("lớp phổ biến đồng nhất, lớp hiếm dị biệt (giống pool tài liệu thật)"
           if regime == "R1" else "độ trải rộng không tương quan với tần suất (trung lập)"))
    print(f"{'='*106}")
    print(f"pool {N_POOL:,} trang · {N_SCEN} kịch bản Zipf(1.25) · kb đầu {100*zipf[0]:.1f}% · "
          f"10 kb hiếm {100*zipf[-10:].sum():.1f}% · độ khó thật " +
          " ".join(f"{TIERS[t].value[:1].upper()}={100*(tier==t).mean():.0f}%" for t in range(4)))

    t0 = time.time()
    ccfg = ClusterConfig(k_coarse=64, split_factor=8, max_share=.02, min_cluster_size=60,
                         fit_sample=N_POOL, kmeans_iters=25, max_depth=3,
                         tail_percentile=99., flatten_rounds=0, flatten_fit_cap=N_POOL)
    idx = hierarchical_cluster(X, ccfg, seed=1)
    sp = local_sparsity(X, ref_size=30_000, k=16, seed=1)
    print(f"clustering + rarity: {len(idx.sizes)} cụm, {time.time()-t0:.0f}s · "
          f"cụm lớn nhất {100*max(v for k,v in idx.sizes.items() if k!=TAIL_CLUSTER)/N_POOL:.2f}% pool")

    # --- probe CMCV ---
    pcfg = ProbeConfig(); probe_tiers, npb = {}, 0
    for cid in idx.sizes:
        m = np.where(idx.assign == cid)[0]
        sel = rng.choice(m, probe_size(len(m), pcfg), replace=False)
        probe_tiers[cid] = [TIERS[t] for t in tier[sel]]; npb += len(sel)
    stats = cluster_weights(idx.assign, probe_tiers, "text", pcfg, alpha=.15)
    dropped = [c for c, s in stats.items() if s.dropped]
    err = []
    for cid, s in stats.items():
        m = idx.assign == cid
        err.append(np.abs(s.p - np.array([(tier[m] == t).mean() for t in range(4)])).max())
    print(f"probe CMCV: {npb:,} trang = {100*npb/N_POOL:.2f}% pool · loại {len(dropped)} cụm rác "
          f"({sum(idx.sizes[c] for c in dropped):,} trang) · |p̂-p|max trung vị {np.median(err):.3f}")

    # chẩn đoán tự chọn cường độ rarity từ chính nhãn probe (không tốn thêm model)
    pidx = np.concatenate([np.where(idx.assign == c)[0][:len(probe_tiers[c])] for c in probe_tiers])
    ptl = [t for c in probe_tiers for t in probe_tiers[c]]
    diag = rarity_regime_check(sp, pidx, ptl)
    rar = cluster_rarity(sp, idx.assign, tau=diag["tau_khuyen_nghi"])
    print(f"chẩn đoán rarity: rho(độ thưa, Medium/Hard) = {diag['rho']:+.3f} "
          f"-> tau tự chọn = {diag['tau_khuyen_nghi']:.1f} "
          f"({'BẬT' if diag['tau_khuyen_nghi'] > 0 else 'TẮT'})")

    valid = ~np.isin(idx.assign, dropped)
    avail, members = defaultdict(int), defaultdict(list)
    for i in np.where(valid)[0]:
        c = (int(idx.assign[i]), TIERS[tier[i]]); avail[c] += 1; members[c].append(i)
    avail = dict(avail)

    scfg = SamplerConfig(alpha=.15, floor_per_cell=8, cap_ratio=.6)
    flat = {"text": (1., 1., 1.)}
    runs = {
        "A. Uniform": rng.choice(N_POOL, BUDGET, replace=False),
        "B. Cluster-only": np.array(draw(allocate_nested(avail, "text", scfg, flat,
                                    idx.paths, budget=BUDGET), members, seed=3), int),
        "C. DDAS (cụm x khó)": np.array(draw(allocate_nested(avail, "text", scfg, pcfg.gain,
                                    idx.paths, budget=BUDGET), members, seed=3), int),
        "D. DDAS + rarity (auto)": np.array(draw(allocate_nested(avail, "text", scfg, pcfg.gain,
                                    idx.paths, budget=BUDGET, cluster_prior=rar), members, seed=3), int),
    }
    rows = {k: evaluate(v, scen, tier) for k, v in runs.items()}

    hdr = (f"{'chiến lược':<24}{'#kb hữu hiệu':>14}{'phủ':>8}{'%kb đầu':>10}"
           f"{'%10kb hiếm':>12}{'%rác':>7}{'E/M/H (%)':>18}{'tín hiệu':>11}")
    print("-" * len(hdr)); print(hdr); print("-" * len(hdr))
    for k, r in rows.items():
        print(f"{k:<24}{r['eff']:>14.1f}{r['cover']:>6}/24{r['head']:>10.1f}{r['tail']:>12.1f}"
              f"{r['waste']:>7.1f}{r['mix'][0]:>8.0f}/{r['mix'][1]:.0f}/{r['mix'][2]:.0f}{r['signal']:>13.3f}")
    base = rows["A. Uniform"]
    print("-" * len(hdr))
    for k, r in list(rows.items())[1:]:
        print(f"  {k:<22} vs Uniform:  đa dạng x{r['eff']/base['eff']:.2f} · "
              f"kb hiếm x{r['tail']/base['tail']:.2f} · kb đầu x{r['head']/base['head']:.2f} · "
              f"rác {base['waste']:.1f}→{r['waste']:.1f}% · tín hiệu x{r['signal']/base['signal']:.2f}")
    return rows


if __name__ == "__main__":
    r1 = run_regime("R1")
    r2 = run_regime("R2")
    print(f"\n{'='*106}\nKẾT LUẬN\n{'='*106}")
    for name in ("C. DDAS (cụm x khó)", "D. DDAS + rarity (auto)"):
        a, b = r1[name], r2[name]
        print(f"{name:<24} R1: tín hiệu x{a['signal']/r1['A. Uniform']['signal']:.2f}, "
              f"kb hiếm x{a['tail']/r1['A. Uniform']['tail']:.2f}  |  "
              f"R2: tín hiệu x{b['signal']/r2['A. Uniform']['signal']:.2f}, "
              f"kb hiếm x{b['tail']/r2['A. Uniform']['tail']:.2f}")
