"""Preflight — kiểm một trang THẬT qua từng model trước khi chạy đợt tốn tiền.

    python3 -m ddas.testkit.preflight --pages data/vietnamese_sample/invoices

Đây là tiêu chí nghiệm thu mốc M0. Chạy nó TRƯỚC mỗi đợt lớn, vì ba loại lỗi
dưới đây chỉ lộ ra khi gọi thật, và nếu để lọt thì chúng hỏng âm thầm chứ
không crash:

  1. SHAPE RESPONSE KHÁC dự đoán -> extract_fn trả rỗng -> trang bị cách ly
     hàng loạt (Paddle), hoặc ParseResult mất bbox (Mistral, xem strict_layout).
  2. HỆ TOẠ ĐỘ SAI -> bbox vẫn là số hợp lệ, IoU vẫn tính ra, nhưng vô nghĩa.
     Preflight in ra bbox thật cạnh kích thước ảnh để nhìn bằng mắt.
  3. MODEL TỰ "SỬA HỘ" nội dung -> hai model đồng thuận ở thứ không có trong
     ảnh. Không tự phát hiện được; preflight in nguyên văn để soát tay.

Preflight KHÔNG thay thế calibrate_tau() (mốc M1). Nó chỉ trả lời "đường ống có
thông không", không trả lời "đồng thuận có nghĩa là đúng không".
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from typing import Dict, List

from ..clients import ClientError, build_clients, env_report
from ..clients.normalize import is_empty_result


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Kiểm một trang thật qua từng model")
    ap.add_argument("--pages", required=True, help="thư mục ảnh/PDF scan")
    ap.add_argument("--page-id", default=None, help="page_id cụ thể (mặc định: trang đầu)")
    ap.add_argument("--cache-dir", default=".cache/ddas")
    ap.add_argument("--no-cache", action="store_true",
                    help="tắt cache để buộc gọi mạng thật (preflight nên dùng)")
    ap.add_argument("--max-side", type=int, default=1600)
    ap.add_argument("--verbose", "-v", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    print("== cấu hình môi trường ==")
    env = env_report()
    for k, ok in env.items():
        print(f"  {'✓' if ok else '✗'} {k}")
    if not any(env.values()):
        print("\nKhông có vai nào được cấu hình. Đặt biến môi trường rồi chạy lại:\n"
              "  export QWEN_BASE_URL=http://<vllm-host>:8000/v1\n"
              "  export MISTRAL_API_KEY=...\n"
              "  export PADDLE_VL_URL=http://<paddle-host>:8080/layout-parsing\n"
              "  export OPENAI_API_KEY=...      # trọng tài §3.3\n"
              "  export GEMINI_API_KEY=...      # pre-annotation §3.3")
        return 2

    clients = build_clients(pages_root=a.pages, cache_dir=a.cache_dir,
                            enable_cache=not a.no_cache, max_side=a.max_side)
    ids = clients.page_store.discover()
    if not ids:
        print(f"\nKhông tìm thấy trang nào trong {a.pages}")
        return 2
    page_id = a.page_id or ids[0]
    img = clients.image_fn(page_id)
    print(f"\n== trang thử: {page_id}  ({img.size[0]}x{img.size[1]} px sau resize) ==")

    ok_all = True
    for name, runner in clients.runners.items():
        print(f"\n-- {name} --")
        t0 = time.monotonic()
        try:
            pr = runner(page_id)
        except ClientError as e:
            ok_all = False
            print(f"  ✗ LỖI: {e}")
            continue
        dt = time.monotonic() - t0
        print(f"  ✓ {dt:.1f}s · {len(pr.boxes)} box · {len(pr.text)} ký tự text · "
              f"{len(pr.formulas)} công thức · {len(pr.tables)} bảng")
        if len(pr.boxes) == 0:
            ok_all = False
            print("  ✗ KHÔNG CÓ BBOX -> layout_sim với model này luôn = 0, "
                  "subtask 'layout' của CMCV mất model này. Xem mistral_ocr.py::strict_layout.")
        else:
            b = pr.boxes
            W, H = img.size
            inside = bool(b[:, 2].max() <= W + 1 and b[:, 3].max() <= H + 1)
            print(f"    bbox x:[{b[:,0].min():.0f},{b[:,2].max():.0f}] "
                  f"y:[{b[:,1].min():.0f},{b[:,3].max():.0f}] "
                  f"{'✓ trong ảnh' if inside else '✗ VƯỢT KHUNG ẢNH — sai hệ toạ độ'}")
            if not inside:
                ok_all = False
            print(f"    nhãn: {sorted(set(pr.labels))}")
        if is_empty_result(pr):
            ok_all = False
            print("  ✗ ParseResult RỖNG")
        print(f"    text (200 ký tự đầu, SOÁT DẤU TIẾNG VIỆT BẰNG MẮT):\n"
              f"      {pr.text[:200]!r}")

    if clients.judge_fn is not None:
        print("\n-- trọng tài §3.3 (gpt-5) --")
        try:
            v = clients.judge_fn(img, None, "E = mc^2", "formula")
            print(f"  ✓ has_error={v.has_error} confidence={v.confidence} "
                  f"note={v.note[:120]!r}")
        except ClientError as e:
            ok_all = False
            print(f"  ✗ LỖI: {e}")

    if clients.preannot_fn is not None:
        print("\n-- pre-annotation §3.3 (gemini-3-pro) --")
        try:
            out = clients.preannot_fn(img, "text")
            print(f"  ✓ {len(out)} ký tự: {out[:160]!r}")
        except ClientError as e:
            ok_all = False
            print(f"  ✗ LỖI: {e}")

    print("\n== số liệu (đối chiếu costmodel.py ở mốc M2) ==")
    print(json.dumps(clients.stats(), indent=2, ensure_ascii=False))
    q = clients.quarantined()
    if q:
        print(f"\n== trang bị cách ly: {len(q)} ==")
        for pid, reason in list(q.items())[:5]:
            print(f"  {pid}: {reason}")

    print("\n" + ("✓ PREFLIGHT ĐẠT — đường ống thông. Bước tiếp: M1, dev-set + calibrate_tau()."
                  if ok_all else
                  "✗ PREFLIGHT KHÔNG ĐẠT — sửa các mục ✗ ở trên TRƯỚC khi chạy đợt lớn."))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
