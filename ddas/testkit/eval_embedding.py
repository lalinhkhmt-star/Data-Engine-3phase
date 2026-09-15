#!/usr/bin/env python3
"""Đo bằng dữ liệu thật: ViT-base thuần, layout-prior thuần, hay ghép cả hai —
cái nào phân cụm đúng loại trang tài liệu nhất?

KHÔNG tin cấu hình mặc định. Đây là công cụ để tự kiểm chứng, đúng nguyên tắc
"chẩn đoán trước khi bật" đã dùng ở density.rarity_regime_check().

=== Chuẩn bị dữ liệu ===

Cách 1 — thư mục theo nhãn (khuyến nghị, cho purity/NMI có giám sát):

    data/
      paper_1cot/        *.pdf   (hoặc *.png, *.jpg đã render sẵn)
      bao_cao_da_cot/     *.pdf
      bang_long_nhau/     *.pdf
      cong_thuc_day_dac/  *.pdf
      scan_viet_tay/      *.pdf
      ...

  Mỗi thư mục con là một LOẠI TRANG bạn biết trước (ground truth), không phải
  nhãn model sinh ra. Muốn đo có ý nghĩa thì cần >= 4 loại, mỗi loại >= 20 trang,
  và các loại phải thực sự khác nhau về cấu trúc (đây là thứ DDAS cần tách được).

  Chạy:
    python3 -m ddas.testkit.eval_embedding --data-dir data/ --dpi 150

Cách 2 — thư mục phẳng không nhãn (chỉ đo silhouette, không đo được purity):

    python3 -m ddas.testkit.eval_embedding --data-dir data_khong_nhan/ --unsupervised

=== Output ===

Bảng so sánh 3 candidate (vit / layout / vit+layout) trên purity, NMI, và
silhouette, kèm ma trận nhầm lẫn cụm-nhãn để soi bằng mắt cụm nào bị trộn.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ddas.embed_real import ViTPageEncoder, combine_features, fit_pca, l2_normalize, render_pdf_page
from ddas.layout_prior import LAYOUT_DIM, extract_layout_prior

IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
PDF_EXT = {".pdf"}


# --------------------------------------------------------------- dữ liệu ----

def collect_dataset(data_dir: Path, max_per_class: int, page_no: int
                    ) -> Tuple[List[Path], List[str], List[bool]]:
    """Quét data_dir. Nếu có thư mục con -> nhãn = tên thư mục. Nếu không -> 1 nhãn 'unlabeled'.

    Trả về (đường dẫn, nhãn, is_pdf) song song theo từng mục.
    """
    subdirs = sorted(p for p in data_dir.iterdir() if p.is_dir())
    paths, labels, is_pdf = [], [], []

    def add(files: List[Path], label: str):
        files = sorted(files)[:max_per_class]
        for f in files:
            paths.append(f); labels.append(label); is_pdf.append(f.suffix.lower() in PDF_EXT)

    if subdirs:
        for sd in subdirs:
            files = [f for f in sd.iterdir() if f.suffix.lower() in (IMG_EXT | PDF_EXT)]
            add(files, sd.name)
    else:
        files = [f for f in data_dir.iterdir() if f.suffix.lower() in (IMG_EXT | PDF_EXT)]
        add(files, "unlabeled")

    return paths, labels, is_pdf


def load_image(path: Path, is_pdf: bool, page_no: int, dpi: int):
    from PIL import Image
    if is_pdf:
        return render_pdf_page(str(path), page_no=page_no, dpi=dpi)
    return Image.open(path).convert("RGB")


# ------------------------------------------------------------- đặc trưng ----

def build_features(paths: List[Path], is_pdf: List[bool], page_no: int, dpi: int,
                   vit_model: str, device: str, pca_dim: Optional[int],
                   layout_weight: float, verbose: bool = True
                   ) -> Tuple[Dict[str, np.ndarray], List[int]]:
    """Trả về {candidate_name: (N,d) matrix} + chỉ số các mẫu hợp lệ (layout prior tính được)."""
    t0 = time.time()
    images, layout_vecs, layout_valid = [], [], []
    for i, (p, ispdf) in enumerate(zip(paths, is_pdf)):
        img = load_image(p, ispdf, page_no, dpi)
        images.append(img)
        if ispdf:
            lp = extract_layout_prior(str(p), page_no=page_no)
            layout_vecs.append(lp.vector); layout_valid.append(lp.is_valid)
        else:
            layout_vecs.append(np.zeros(LAYOUT_DIM, np.float32)); layout_valid.append(False)
        if verbose and (i + 1) % 25 == 0:
            print(f"  đã đọc {i+1}/{len(paths)} trang ({time.time()-t0:.0f}s)", file=sys.stderr)

    if verbose:
        print(f"Đọc {len(paths)} trang xong sau {time.time()-t0:.1f}s. "
              f"Layout-prior hợp lệ: {sum(layout_valid)}/{len(paths)} "
              f"(phần còn lại là scan/ảnh không có text layer)", file=sys.stderr)

    t1 = time.time()
    encoder = ViTPageEncoder(model_name=vit_model, device=device)
    vit_raw = encoder.encode(images)
    if verbose:
        print(f"ViT-base ({vit_model}, dim={vit_raw.shape[1]}) encode xong "
              f"sau {time.time()-t1:.1f}s", file=sys.stderr)

    vit_feat = vit_raw
    if pca_dim and pca_dim < vit_raw.shape[1]:
        vit_feat = fit_pca(vit_raw, pca_dim)
        if verbose:
            print(f"PCA {vit_raw.shape[1]} -> {pca_dim} chiều (fit trên chính batch này — "
                  f"xem cảnh báo trong embed_real.fit_pca)", file=sys.stderr)

    layout_mat = np.stack(layout_vecs).astype(np.float32)

    candidates = {
        "vit": l2_normalize(vit_feat),
        "layout": l2_normalize(layout_mat),
        "vit+layout": combine_features(vit_feat, layout_mat,
                                       vit_weight=1.0, layout_weight=layout_weight),
    }
    return candidates, layout_valid


# --------------------------------------------------------------- đo lường ---

def purity(pred: np.ndarray, true_labels: List[str]) -> float:
    n = len(pred)
    total = 0
    for c in np.unique(pred):
        idx = np.where(pred == c)[0]
        cnt = Counter(true_labels[i] for i in idx)
        total += cnt.most_common(1)[0][1]
    return total / n


def confusion(pred: np.ndarray, true_labels: List[str]) -> Dict[str, Dict[int, int]]:
    out: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for p, t in zip(pred, true_labels):
        out[t][int(p)] += 1
    return {k: dict(v) for k, v in out.items()}


def assignments(pred: np.ndarray, true_labels: List[str], paths: List[Path]) -> Dict[str, list]:
    """file -> cluster, gộp theo cluster để dễ soi."""
    by_cluster: Dict[str, list] = defaultdict(list)
    for p, t, path in zip(pred, true_labels, paths):
        by_cluster[str(int(p))].append({"file": f"{path.parent.name}/{path.name}", "true_label": t})
    return dict(sorted(by_cluster.items(), key=lambda kv: int(kv[0])))


def run_kmeans(X: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    from sklearn.cluster import KMeans
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)


def evaluate(candidates: Dict[str, np.ndarray], labels: List[str], k: Optional[int],
            unsupervised: bool, paths: Optional[List[Path]] = None) -> Dict[str, dict]:
    from sklearn.metrics import normalized_mutual_info_score, silhouette_score

    true_k = len(set(labels)) if not unsupervised else (k or 8)
    results = {}
    for name, X in candidates.items():
        pred = run_kmeans(X, true_k)
        row = {"k": true_k}
        try:
            row["silhouette"] = float(silhouette_score(X, pred))
        except Exception:
            row["silhouette"] = float("nan")
        if not unsupervised:
            row["purity"] = purity(pred, labels)
            row["nmi"] = float(normalized_mutual_info_score(labels, pred))
            row["confusion"] = confusion(pred, labels)
        if paths is not None:
            row["assignments"] = assignments(pred, labels, paths)
        results[name] = row
    return results


def print_report(results: Dict[str, dict], unsupervised: bool):
    print("\n" + "=" * 78)
    print("KẾT QUẢ — cao hơn là tốt hơn cho purity/NMI/silhouette")
    print("=" * 78)
    hdr = f"{'candidate':<14}{'k':>5}{'silhouette':>13}"
    if not unsupervised:
        hdr += f"{'purity':>10}{'NMI':>8}"
    print(hdr)
    print("-" * len(hdr))
    for name, r in results.items():
        line = f"{name:<14}{r['k']:>5}{r['silhouette']:>13.3f}"
        if not unsupervised:
            line += f"{r['purity']:>10.3f}{r['nmi']:>8.3f}"
        print(line)

    if not unsupervised:
        best = max(results, key=lambda n: results[n]["nmi"])
        print(f"\n>> NMI cao nhất: {best} ({results[best]['nmi']:.3f})")
        vit_nmi = results.get("vit", {}).get("nmi")
        best_nmi = results[best]["nmi"]
        if vit_nmi is not None and best != "vit" and best_nmi - vit_nmi > 0.03:
            print(f"   ViT thuần ({vit_nmi:.3f}) THUA {best} — nghi ngờ ở câu trả lời "
                  f"trước có cơ sở trên dữ liệu này: DINOv2 không tự tách được cấu trúc "
                  f"bố cục, cần layout-prior.")
        elif vit_nmi is not None and vit_nmi >= best_nmi - 0.01:
            print(f"   ViT thuần ngang hoặc hơn ({vit_nmi:.3f}) — trên dữ liệu này ViT-base "
                  f"đã đủ, layout-prior không cần thiết (hoặc cách trích xuất layout-prior "
                  f"ở đây chưa tốt).")

        print("\nMa trận nhầm lẫn (nhãn thật -> {cụm: số lượng}), candidate tốt nhất:")
        for label, dist in results[best]["confusion"].items():
            top = sorted(dist.items(), key=lambda kv: -kv[1])
            print(f"  {label:<22} " + "  ".join(f"cụm{c}:{n}" for c, n in top))


# --------------------------------------------------------------------- main -

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--page-no", type=int, default=0)
    ap.add_argument("--max-per-class", type=int, default=200)
    ap.add_argument("--vit-model", default="facebook/dinov2-base",
                    help="Model HF bất kỳ có AutoModel/AutoImageProcessor. "
                         "Thử thêm: facebook/dinov2-small (nhanh hơn nhiều trên CPU), "
                         "openai/clip-vit-base-patch32.")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--pca-dim", type=int, default=None,
                    help="Nếu đặt, PCA vector ViT xuống chiều này TRƯỚC khi so sánh "
                         "(fit trên chính batch đánh giá — chỉ hợp lệ để so sánh nội bộ).")
    ap.add_argument("--layout-weight", type=float, default=0.6)
    ap.add_argument("--k", type=int, default=None,
                    help="Số cụm K-Means. Mặc định = số nhãn thật (có giám sát) hoặc 8 (không giám sát).")
    ap.add_argument("--unsupervised", action="store_true",
                    help="data-dir không có thư mục con theo nhãn -> chỉ đo silhouette.")
    ap.add_argument("--out", type=Path, default=None, help="Lưu kết quả JSON.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not args.data_dir.exists():
        print(f"LỖI: không thấy {args.data_dir}", file=sys.stderr); sys.exit(1)

    paths, labels, is_pdf = collect_dataset(args.data_dir, args.max_per_class, args.page_no)
    if not paths:
        print(f"LỖI: không tìm thấy file .pdf/.png/.jpg nào trong {args.data_dir}", file=sys.stderr)
        sys.exit(1)

    n_labels = len(set(labels))
    print(f"Tìm thấy {len(paths)} trang, {n_labels} nhãn: {dict(Counter(labels))}", file=sys.stderr)
    if not args.unsupervised and n_labels < 2:
        print("CẢNH BÁO: chỉ 1 nhãn — không đo được purity/NMI có ý nghĩa. "
              "Thêm --unsupervised nếu đây là chủ ý, hoặc tổ chức data-dir theo thư mục nhãn.",
              file=sys.stderr)

    candidates, layout_valid = build_features(
        paths, is_pdf, args.page_no, args.dpi, args.vit_model, args.device,
        args.pca_dim, args.layout_weight)

    if is_pdf and sum(layout_valid) < len(paths) * 0.5:
        print(f"CẢNH BÁO: chỉ {sum(layout_valid)}/{len(paths)} trang có text layer hợp lệ — "
              f"candidate 'layout' và 'vit+layout' sẽ kém tin cậy trên phần còn lại "
              f"(chúng là trang scan thuần hoặc ảnh, layout-prior = vector 0).", file=sys.stderr)

    results = evaluate(candidates, labels, args.k, args.unsupervised, paths=paths)
    print_report(results, args.unsupervised)

    if args.out:
        args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"\nĐã lưu {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
