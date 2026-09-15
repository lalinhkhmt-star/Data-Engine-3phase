#!/usr/bin/env python3
"""Tải một mẫu từ DocLayNet-v1.2 (HuggingFace) và ghi ra đúng cấu trúc thư mục
mà eval_embedding.py cần: data_dir/<doc_category>/*.pdf

DocLayNet-v1.2 có field `pdf` (bytes PDF THẬT, có text layer — field `pdf_cells`
xác nhận nội dung đã được trích) và `metadata.doc_category` là 1 trong 6 nhãn:
financial_reports, scientific_articles, laws_and_regulations, government_tenders,
manuals, patents. Đây là điều kiện quan trọng nhất để layout_prior.py hoạt động —
không phải mọi dataset layout công khai đều giữ PDF gốc, phần lớn chỉ có ảnh PNG.

CHƯA CHẠY THỬ được trong sandbox này (Python externally-managed, không cài được
`datasets`) — kiểm tra field name khi chạy lần đầu, in ra sample đầu tiên trước
khi tải hàng loạt.

Cài đặt:
    pip install datasets huggingface_hub      # khuyến nghị trong venv riêng

Chạy:
    python3 -m ddas.testkit.fetch_doclaynet_sample --out data/doclaynet --per-category 40
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-category", type=int, default=40,
                    help="Số trang tối đa lấy mỗi doc_category.")
    ap.add_argument("--split", default="train")
    ap.add_argument("--dataset", default="docling-project/DocLayNet-v1.2",
                    help="Đổi sang ds4sd/DocLayNet-v1.1 nếu v1.2 không truy cập được.")
    ap.add_argument("--shuffle-buffer", type=int, default=20_000,
                    help="QUAN TRỌNG: stream của dataset này KHÔNG xáo trộn sẵn — "
                         "200 mẫu đầu tiên đo thực tế đều là financial_reports. "
                         "Không shuffle thì sẽ không bao giờ thấy đủ 6 category "
                         "trong thời gian hợp lý.")
    ap.add_argument("--max-scan", type=int, default=200_000,
                    help="Trần số mẫu quét qua (kể cả bị bỏ vì category đã đủ), "
                         "để không chạy vô hạn nếu 1 category hiếm không đủ số lượng.")
    args = ap.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("LỖI: chưa cài `datasets`. Chạy: pip install datasets huggingface_hub",
              file=sys.stderr)
        sys.exit(1)

    print(f"Mở streaming {args.dataset} split={args.split}, "
          f"shuffle buffer={args.shuffle_buffer} ...", file=sys.stderr)
    ds = load_dataset(args.dataset, split=args.split, streaming=True)
    ds = ds.shuffle(seed=0, buffer_size=args.shuffle_buffer)

    # In thử 1 sample để xác nhận field name khớp trước khi tải hàng loạt —
    # schema dataset public có thể đổi giữa các version, đừng tin mù.
    first = next(iter(ds))
    print("Field có trong 1 sample đầu:", list(first.keys()), file=sys.stderr)
    if "pdf" not in first or "metadata" not in first:
        print("CẢNH BÁO: không thấy field 'pdf' hoặc 'metadata' như tài liệu mô tả — "
              "kiểm tra lại schema trên trang HuggingFace trước khi tiếp tục.", file=sys.stderr)

    args.out.mkdir(parents=True, exist_ok=True)
    counts: Counter = Counter()
    for d in args.out.iterdir():
        if d.is_dir():
            counts[d.name] = len(list(d.glob("*.pdf")))
    if sum(counts.values()):
        print(f"Tiếp tục từ dữ liệu đã có: {dict(counts)}", file=sys.stderr)
    written = 0

    scanned = 0
    for ex in ds:
        scanned += 1
        if scanned > args.max_scan:
            print(f"CẢNH BÁO: chạm --max-scan={args.max_scan} trước khi đủ mẫu mọi category. "
                  f"Kết quả hiện có: {dict(counts)}", file=sys.stderr)
            break
        cat = (ex.get("metadata") or {}).get("doc_category", "unknown")
        if counts[cat] >= args.per_category:
            if len(counts) >= 6 and all(counts[c] >= args.per_category for c in counts):
                break
            continue
        pdf_bytes = ex.get("pdf")
        if not pdf_bytes:
            continue
        d = args.out / cat
        d.mkdir(exist_ok=True)
        page_id = ex.get("metadata", {}).get("page_no", written)
        fp = d / f"{cat}_{counts[cat]:04d}.pdf"
        fp.write_bytes(pdf_bytes if isinstance(pdf_bytes, (bytes, bytearray)) else bytes(pdf_bytes))
        counts[cat] += 1
        written += 1
        if written % 20 == 0:
            print(f"  đã ghi {written} trang, theo category: {dict(counts)}", file=sys.stderr)

    print(f"\nXong. Tổng {written} PDF trong {args.out}, theo category: {dict(counts)}", file=sys.stderr)
    print(f"Chạy tiếp: python3 -m ddas.testkit.eval_embedding --data-dir {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
