"""So sánh hai model trọng tài §3.3 trên CÙNG một đầu vào có ĐÁP ÁN BIẾT TRƯỚC.

Vì sao cần đáp án biết trước: chạy trọng tài trên một bản nháp bất kỳ chỉ cho
biết nó "tìm thấy gì", không cho biết nó BỎ SÓT gì. Mà bỏ sót mới là rủi ro
chết người của §3.3 — trọng tài nhận nhầm là sạch thì nhãn sai chui thẳng vào
tập train (xem IMPLEMENTATION_STATUS.md §3.3 mục 1).

Nên ở đây bản nháp được tạo bằng cách chép đúng bảng trong ảnh rồi CHÈN 4 lỗi
đã biết. Chấm mỗi model theo hai chiều:
  - bắt được mấy / 4 lỗi đã chèn  (càng cao càng tốt)
  - báo thêm bao nhiêu thứ khác   (cần người đọc: lỗi thật OCR bỏ sót, hay
                                   bắt bẻ vụn vặt kiểu thừa/thiếu khoảng trắng)

    OPENAI_API_KEY=... python3 -m ddas.testkit.ab_judge_models
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv as _load_dotenv
_load_dotenv()

from ..clients.cache import DiskCache
from ..clients.openai_compat import OpenAICompatClient, OpenAICompatConfig
from ..judge_refine import make_judge_fn
from ..prompts import VERDICT_SCHEMA
from ..render import render_label

IMAGE = "tmp/crops_test2/crop_00_table.jpg"     # bảng 12 dây thần kinh sọ, scan thật
MODELS = ["gpt-5", "gpt-5-mini"]

# Bản nháp: chép đúng bảng trong ảnh, TRỪ 4 chỗ cố ý sai bên dưới.
DRAFT = """<table>
<tr><th>Số</th><th>TÊN GỌI THẦN KINH SỌ</th><th>LỖ RA KHỎI HỘP SỌ</th><th>LÂM SÀNG</th></tr>
<tr><td>I</td><td>dải (hay thần kinh) khứu</td><td>mảnh sàng</td><td>mắt khứu giác (mắt mùi)</td></tr>
<tr><td>II</td><td>thần kinh thị</td><td>ống thị giác</td><td>nhìn rõ, tinh tường</td></tr>
<tr><td>III</td><td>thần kinh vận nhãn (hay vận nhãn chung)</td><td>khe ổ mắt trên hay khe bướm</td><td>sụp mi, giãn đồng tử, lé mắt ngoài</td></tr>
<tr><td>IV</td><td>thần kinh ròng rọc</td><td>khe ổ mắt trên</td><td>nhìn đôi</td></tr>
<tr><td>V</td><td>thần kinh sinh ba - thần kinh mắt Willis - thần kinh hàm trên - thần kinh hàm dưới</td><td>khe ổ mắt trên, lỗ tròn (lỗ tròn lớn), lỗ tròn</td><td>đau thần kinh V ở mặt (trán, lệ, mũi), đau thần kinh V ở mặt, co thắt nửa mặt /các cơ nhai</td></tr>
<tr><td>VI</td><td>thần kinh vận nhãn ngoài</td><td>khe ổ mắt trên</td><td>lé mắt trong</td></tr>
<tr><td>VII</td><td>thần kinh mặt và TK trung gian (VII bis)</td><td>lỗ ống tai trong (CAI)</td><td>liệt mặt ngoại biên, co thắt nửa mặt /mi mắt</td></tr>
<tr><td>VIII</td><td>thần kinh tiền đình-ốc tai (TK thính giác và thần kinh tiền đình)</td><td>lỗ ống tai trong</td><td>điếc một bên, chóng mặt, ù tai</td></tr>
<tr><td>IX</td><td>thần kinh thiệt hầu</td><td>lỗ cảnh</td><td>rối loạn vị giác</td></tr>
<tr><td>X</td><td>thần kinh lang thang (TK phế vị)</td><td>lỗ cảnh</td><td>khẩu cái mềm</td></tr>
<tr><td>XII</td><td>thần kinh hạ thiệt</td><td>ống hạ thiệt</td><td>teo nửa lưỡi</td></tr>
</table>"""

# Đáp án cho 4 lỗi đã chèn. Chấm trên BẢN SỬA (`corrected`), không phải trên
# phần model kể lể — nhãn đi vào tập train là `corrected`, model "có nhắc tới"
# mà không sửa thì vẫn là nhãn sai.
#
# CẢNH BÁO đã mắc một lần: chấm bằng "từ khoá xuất hiện ở đâu đó" cho kết quả
# SAI — chuỗi sai ('mắt khứu giác') cũng chứa từ khoá, nên model để nguyên lỗi
# vẫn bị tính là bắt được. Phải kiểm cả hai chiều: bản đúng CÓ mặt và bản sai
# KHÔNG còn.
INJECTED = [
    {"id": "I-dấu", "mô tả": "Hàng I: 'mất khứu giác (mất mùi)' -> 'mắt ... (mắt mùi)' — sai dấu",
     "must": "mất khứu giác", "must_not": "mắt khứu giác"},
    {"id": "II-nghĩa", "mô tả": "Hàng II: 'nhìn mờ, mù' -> 'nhìn rõ, tinh tường' — ngược nghĩa",
     "must": "nhìn mờ", "must_not": "tinh tường"},
    {"id": "V-giải phẫu", "mô tả": "Hàng V hàm dưới: 'lỗ bầu dục' -> 'lỗ tròn'",
     "must": "bầu dục", "must_not": None},
    {"id": "XI-thiếu hàng", "mô tả": "Thiếu hẳn hàng XI (thần kinh phụ / lỗ cảnh / teo cơ thang)",
     "must": "thần kinh phụ", "must_not": None},
]


def _fixed(corrected: str, item) -> bool:
    """Lỗi này đã được SỬA ĐÚNG trong bản `corrected` chưa."""
    c = (corrected or "").lower()
    if item["must"].lower() not in c:
        return False
    return item["must_not"] is None or item["must_not"].lower() not in c


def main() -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print("THIẾU OPENAI_API_KEY (đặt trong .env hoặc export).")
        return 2
    img_path = Path(IMAGE)
    if not img_path.exists():
        print(f"Không thấy ảnh: {img_path}")
        return 2

    from PIL import Image
    page = Image.open(img_path)
    image_fn = lambda _pid: page

    shot = render_label(DRAFT, "table", 150)
    print(f"ảnh gốc {page.size} · ảnh render {shot.image.size if shot.ok else 'LỖI'}"
          f" (backend={shot.backend})\n")

    results = {}
    for model in MODELS:
        client = OpenAICompatClient(
            OpenAICompatConfig(model=model,
                               base_url=os.environ.get("OPENAI_BASE_URL",
                                                       "https://api.openai.com/v1"),
                               max_tokens=24000, max_tokens_param="max_completion_tokens",
                               temperature=None,
                               json_schema={"name": "judge_verdict",
                                            "schema": VERDICT_SCHEMA}),
            DiskCache(".cache/ddas_ab_judge"))
        judge_fn = make_judge_fn(client.as_call_model("judge-v1"))

        print(f"── {model} … ", end="", flush=True)
        t0 = time.monotonic()
        v = judge_fn(page, shot.image if shot.ok else None, DRAFT, "table")
        dt = time.monotonic() - t0
        s = client.stats.summary()
        print(f"{dt:.0f}s · vào {s.get('tokens_in', 0):,} ra {s.get('tokens_out', 0):,} token")

        found = [x["id"] for x in INJECTED if _fixed(v.corrected, x)]
        results[model] = {"verdict": v, "found": found, "sec": dt, "stats": s}

    # ---------------------------------------------------------- báo cáo ----
    print("\n" + "=" * 74)
    print(f"{'lỗi đã chèn (chấm trên bản sửa)':<34}" + "".join(f"{m:>18}" for m in MODELS))
    print("-" * 74)
    for x in INJECTED:
        row = f"{x['id']:<34}"
        for m in MODELS:
            row += f"{'SỬA ĐÚNG' if x['id'] in results[m]['found'] else '— BỎ SÓT':>18}"
        print(row)
    print("-" * 74)
    print(f"{'TỔNG bắt được / 4':<34}" +
          "".join(f"{len(results[m]['found']):>18}" for m in MODELS))
    print(f"{'has_error':<34}" +
          "".join(f"{str(results[m]['verdict'].has_error):>18}" for m in MODELS))
    print(f"{'confidence':<34}" +
          "".join(f"{results[m]['verdict'].confidence:>18.2f}" for m in MODELS))
    print(f"{'có đề xuất bản sửa':<34}" +
          "".join(f"{str(results[m]['verdict'].corrected is not None):>18}" for m in MODELS))
    print(f"{'thời gian (giây)':<34}" +
          "".join(f"{results[m]['sec']:>18.0f}" for m in MODELS))
    print(f"{'token ra':<34}" +
          "".join(f"{results[m]['stats'].get('tokens_out', 0):>18,}" for m in MODELS))

    for m in MODELS:
        print(f"\n===== {m} · ghi chú khoanh lỗi =====")
        print((results[m]["verdict"].note or "(rỗng)")[:1400])

    out = Path(".cache/ab_judge_result.json")
    out.write_text(json.dumps({m: {
        "has_error": r["verdict"].has_error, "confidence": r["verdict"].confidence,
        "note": r["verdict"].note, "corrected": r["verdict"].corrected,
        "found": r["found"], "sec": r["sec"], "stats": r["stats"]}
        for m, r in results.items()}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n(đã lưu {out})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
