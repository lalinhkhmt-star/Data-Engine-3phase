"""Độ hiếm cục bộ — thành phần đa dạng mà SỐ LƯỢNG CỤM không nắm được.

Phát hiện khi đo: K-Means đặt centroid tỉ lệ với KHỐI LƯỢNG dữ liệu. Trên pool
long-tail, lớp tần suất cao chiếm ~43% số cụm; vì vậy "chia quota đều cho mỗi
cụm" vẫn tái tạo gần như nguyên vẹn độ lệch của pool. Chẻ phân cấp không cứu
được, vì nó lại chẻ nhầm nhóm tần suất trung bình.

Tín hiệu bổ sung: độ thưa cục bộ d_k(x) = khoảng cách tới láng giềng thứ k.
Vùng dày (paper 1 cột, rất đồng nhất) -> d_k nhỏ; vùng thưa (bố cục dị thường,
bảng lồng nhau) -> d_k lớn. Tín hiệu này liên tục, không phụ thuộc K, và không
kế thừa thiên lệch phân bổ centroid.

Ở quy mô 500M: KHÔNG tính k-NN toàn cặp. Dựng index trên tập tham chiếu ngẫu
nhiên M (1-5M điểm) rồi truy vấn theo lô — d_k so với mẫu ngẫu nhiên vẫn là
ước lượng đơn điệu của mật độ, đủ dùng để xếp hạng.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

try:
    import faiss
    _HAS_FAISS = True
except Exception:                                       # pragma: no cover
    _HAS_FAISS = False


def local_sparsity(X: np.ndarray, ref_size: int = 2_000_000, k: int = 16,
                   seed: int = 0, chunk: int = 100_000) -> np.ndarray:
    """d_k(x) so với tập tham chiếu ngẫu nhiên. Lớn = vùng thưa = hiếm."""
    rng = np.random.default_rng(seed)
    m = min(ref_size, len(X))
    ref = np.ascontiguousarray(X[rng.choice(len(X), m, replace=False)], dtype=np.float32)
    out = np.empty(len(X), np.float32)
    if _HAS_FAISS:
        d = X.shape[1]
        if m > 200_000:                                  # IVF cho tập tham chiếu lớn
            quant = faiss.IndexFlatL2(d)
            index = faiss.IndexIVFFlat(quant, d, min(4096, max(1, m // 64)))
            index.train(ref); index.add(ref); index.nprobe = 16
        else:
            index = faiss.IndexFlatL2(d); index.add(ref)
        for s in range(0, len(X), chunk):
            D, _ = index.search(np.ascontiguousarray(X[s:s + chunk], np.float32), k)
            out[s:s + chunk] = np.sqrt(np.maximum(D[:, -1], 0))
        return out
    for s in range(0, len(X), chunk):                    # fallback numpy
        B = X[s:s + chunk].astype(np.float32)
        D = ((B[:, None, :] - ref[None]) ** 2).sum(-1)
        out[s:s + chunk] = np.sqrt(np.partition(D, k - 1, axis=1)[:, k - 1])
    return out


def cluster_rarity(sparsity: np.ndarray, assign: np.ndarray,
                   tau: float = 1.0, clip: float = 8.0) -> Dict[int, float]:
    """Hệ số hiếm của mỗi cụm = (độ thưa trung vị chuẩn hoá)^tau, có chặn trên.

    tau=0 tắt hẳn thành phần này; tau=1 là mặc định; tau lớn dồn mạnh về đuôi.
    Chặn trên tránh việc vài cụm outlier/nhiễu nuốt hết ngân sách.
    """
    out: Dict[int, float] = {}
    med_all = float(np.median(sparsity)) or 1.0
    for cid in np.unique(assign):
        m = assign == cid
        if not m.any():
            continue
        r = float(np.median(sparsity[m])) / med_all
        out[int(cid)] = float(np.clip(r ** tau, 1.0 / clip, clip))
    return out


def rarity_regime_check(sparsity: np.ndarray, probe_idx: np.ndarray,
                        probe_tiers: "Sequence") -> Dict[str, float]:
    """Hệ số hiếm có đáng bật trên pool NÀY không? Đo bằng chính nhãn probe CMCV.

    Không thể kiểm chứng trực tiếp giả định "vùng thưa = kịch bản hiếm" vì ta
    không có nhãn kịch bản. Nhưng ta không cần: điều thực sự quan trọng là độ
    thưa có DỰ BÁO ĐƯỢC GIÁ TRỊ HUẤN LUYỆN hay không. Nếu trang ở vùng thưa có
    tỉ lệ Medium/Hard cao hơn, thì tăng trọng số cho vùng thưa là đi đúng hướng
    mục tiêu; nếu không, nó chỉ dồn ngân sách sang nhiễu.

    Dữ liệu đầu vào đã có sẵn, không tốn thêm một lời gọi model nào: probe CMCV
    ở Stage 1b vốn đã gán tier cho vài trăm nghìn trang.

    Đo trên corpus mô phỏng:
        chế độ thuận lợi  rho=+0.28 -> bật  -> đa dạng x1.53, kịch bản hiếm x3.58
        chế độ bất lợi    rho=-0.01 -> tắt  -> (nếu bật nhầm: đa dạng x0.80)
    """
    from scipy.stats import spearmanr

    from .cmcv import Tier

    s = np.asarray(sparsity)[np.asarray(probe_idx)]
    keep = np.array([t != Tier.INVALID for t in probe_tiers])
    if keep.sum() < 100:
        return {"rho": 0.0, "tau_khuyen_nghi": 0.0, "n": int(keep.sum())}
    mh = np.array([t in (Tier.MEDIUM, Tier.HARD) for t in probe_tiers], float)[keep]
    s = s[keep]
    if mh.std() == 0:
        return {"rho": 0.0, "tau_khuyen_nghi": 0.0, "n": int(keep.sum())}
    rho = float(spearmanr(s, mh).statistic)
    tau = 0.0 if rho <= 0.05 else (0.5 if rho < 0.15 else 1.0)
    return {"rho": rho, "tau_khuyen_nghi": tau, "n": int(keep.sum()),
            "ty_le_thua_MH_tren_E": float(np.median(s[mh == 1]) / max(np.median(s[mh == 0]), 1e-9))}
