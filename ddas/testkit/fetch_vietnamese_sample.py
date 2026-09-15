#!/usr/bin/env python3
"""Tải một mẫu ảnh tài liệu tiếng Việt THẬT (không synthetic) từ HuggingFace,
ghi ra đúng cấu trúc thư mục mà eval_embedding.py cần: data_dir/<category>/*.jpg

3 category tìm được đủ sạch và đủ lớn (xem EMBEDDING_EVAL_REPORT.md để biết
quá trình loại các nguồn khác — CCCD/CMND bị loại vì là ảnh giấy tờ tùy thân thật
của người dân, nhiều khả năng thu thập không rõ có đồng ý; nhiều dataset OCR khác
là ảnh synthetic hoặc crop dòng chữ viết tay, không phải ảnh cả trang):

  invoices  — N9h1ax/Vietnamese_invoices (Roboflow, ảnh chụp hóa đơn VN thật,
              apache-2.0). Có augmentation trùng lặp (rotate/flip) do xuất từ
              Roboflow — script này lọc theo tên file gốc trước hậu tố `.rf.<hash>`
              để không lấy nhiều bản sao của cùng 1 ảnh.
  receipts  — MC-OCR (cuộc thi RIVF2021, mirror DThai/mcocr-test-1), ảnh chụp
              biên lai VN thật bằng điện thoại.
  theses    — hydroshiba/hcmus-doc-layout, thư mục images/val/ — ảnh scan THẬT
              (không lẫn trang synthetic, tác giả README đã tách riêng) của
              luận văn đại học HCMUS. License repo ghi "other" — dùng cho mục
              đích kiểm tra nội bộ, không phải để phân phối lại; kiểm tra kỹ
              nếu định dùng cho việc khác ngoài test này.

CHỈ 3 category — ít hơn 6 category của DocLayNet — nên purity/NMI đo được sẽ
dễ cao hơn giả tạo (bài toán 3 lớp dễ hơn 6 lớp). Đủ để kiểm tra nhanh xem
model có tách được layout tiếng Việt hay không, KHÔNG đủ để kết luận chắc chắn.

Cài đặt:
    pip install datasets huggingface_hub pillow

Chạy:
    python3 -m ddas.testkit.fetch_vietnamese_sample --out data/vietnamese_sample --per-category 60
"""
from __future__ import annotations

import argparse
import io
import re
import sys
from pathlib import Path


def fetch_invoices(out_dir: Path, n: int) -> int:
    """N9h1ax/Vietnamese_invoices — imagefolder trên HF hub, tải trực tiếp qua
    huggingface_hub thay vì `datasets` vì đây là ảnh thô không kèm label parquet."""
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    files = api.list_repo_files("N9h1ax/Vietnamese_invoices", repo_type="dataset")
    imgs = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    # Roboflow augment: "<ten_goc>.rf.<hash>.jpg" — nhiều bản augment của cùng
    # 1 ảnh gốc. Lọc theo tên gốc để tránh eval_embedding.py học nhầm chính nó.
    seen_base = set()
    picked = []
    for f in sorted(imgs):
        base = re.sub(r"\.rf\.[0-9a-f]+\.(jpg|jpeg|png)$", "", f.split("/")[-1], flags=re.I)
        if base in seen_base:
            continue
        seen_base.add(base)
        picked.append(f)
        if len(picked) >= n:
            break

    d = out_dir / "invoices"
    d.mkdir(parents=True, exist_ok=True)
    written = 0
    for f in picked:
        local = hf_hub_download("N9h1ax/Vietnamese_invoices", f, repo_type="dataset")
        ext = Path(f).suffix.lower()
        dst = d / f"invoices_{written:04d}{ext}"
        dst.write_bytes(Path(local).read_bytes())
        written += 1
        if written % 20 == 0:
            print(f"  invoices: {written}/{len(picked)}", file=sys.stderr)
    return written


def fetch_receipts(out_dir: Path, n: int) -> int:
    """MC-OCR (mirror DThai/mcocr-test-1) — stream để không tải hết bản đầy đủ
    (~54 GB); dataset có field `image` (PIL) trực tiếp, không cần giải nén."""
    from datasets import load_dataset

    ds = load_dataset("DThai/mcocr-test-1", split="train", streaming=True)
    d = out_dir / "receipts"
    d.mkdir(parents=True, exist_ok=True)
    written = 0
    for ex in ds:
        img = ex.get("image")
        if img is None:
            continue
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(d / f"receipts_{written:04d}.jpg", "JPEG", quality=90)
        written += 1
        if written % 20 == 0:
            print(f"  receipts: {written}/{n}", file=sys.stderr)
        if written >= n:
            break
    return written


def fetch_theses(out_dir: Path, n: int) -> int:
    """hydroshiba/hcmus-doc-layout, images/val/ — 916 ảnh THẬT (không synthetic,
    tác giả cố ý tách val ra real-only để đo đúng khả năng đọc tài liệu thật)."""
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    files = api.list_repo_files("hydroshiba/hcmus-doc-layout", repo_type="dataset")
    imgs = sorted(f for f in files if f.startswith("images/val/") and f.lower().endswith(".jpg"))

    d = out_dir / "theses"
    d.mkdir(parents=True, exist_ok=True)
    written = 0
    for f in imgs[:n]:
        local = hf_hub_download("hydroshiba/hcmus-doc-layout", f, repo_type="dataset")
        dst = d / f"theses_{written:04d}.jpg"
        dst.write_bytes(Path(local).read_bytes())
        written += 1
        if written % 20 == 0:
            print(f"  theses: {written}/{min(n, len(imgs))}", file=sys.stderr)
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-category", type=int, default=60,
                    help="Số ảnh tối đa lấy mỗi category.")
    args = ap.parse_args()

    try:
        import datasets  # noqa: F401
        import huggingface_hub  # noqa: F401
    except ImportError:
        print("LỖI: chưa cài `datasets`/`huggingface_hub`. "
              "Chạy: pip install datasets huggingface_hub pillow", file=sys.stderr)
        sys.exit(1)

    args.out.mkdir(parents=True, exist_ok=True)

    print("Tải invoices (N9h1ax/Vietnamese_invoices) ...", file=sys.stderr)
    n_inv = fetch_invoices(args.out, args.per_category)

    print("Tải receipts (MC-OCR / DThai/mcocr-test-1, streaming) ...", file=sys.stderr)
    n_rec = fetch_receipts(args.out, args.per_category)

    print("Tải theses (hydroshiba/hcmus-doc-layout, images/val real-only) ...", file=sys.stderr)
    n_th = fetch_theses(args.out, args.per_category)

    print(f"\nXong. invoices={n_inv}, receipts={n_rec}, theses={n_th}, "
          f"tổng {n_inv + n_rec + n_th} ảnh trong {args.out}", file=sys.stderr)
    print(f"Chạy tiếp: python3 -m ddas.testkit.eval_embedding --data-dir {args.out} "
          f"--vit-model openai/clip-vit-base-patch32", file=sys.stderr)


if __name__ == "__main__":
    main()
