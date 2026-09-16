"""Chạy §3.3 (Judge-and-Refine) THẬT trên vài mẫu tự nhập — không cần CMCV/GPU.

Dùng khi bạn CHỦ Ý chưa dựng Qwen3-VL/PaddleOCR-VL/Mistral và chỉ muốn trả lời
câu hỏi cốt lõi của §3.3: "render-then-verify có thật sự giúp GPT-5 phát hiện
lỗi cấu trúc không?" — không cần đợi cả pipeline §3.1/§3.2 chạy được.

Cần:
  export OPENAI_API_KEY=...
  (khuyến nghị) sudo apt install texlive-latex-base texlive-latex-extra
                thiếu pdflatex -> rơi về mathtext, KHÔNG phải LaTeX đầy đủ,
                \\begin{bmatrix}/\\begin{cases}/... render lỗi dù công thức đúng

Cách dùng — sửa DATASET bên dưới trỏ vào ảnh thật + nháp SAI của bạn (nháp
càng gần thật càng đo đúng: lấy từ OCR nào đó bạn có sẵn, tự gõ sai một chỗ,
hoặc copy nguyên bản đúng để đo "false-clean" — tỉ lệ trọng tài nhận nhầm là
sạch), rồi chạy:

    python3 -m ddas.testkit.run_judge_refine_real
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Nạp .env NGAY ĐẦU module — trước khi bất kỳ chỗ nào đọc os.environ. Biến đã
# export sẵn trong shell luôn thắng (python-dotenv mặc định override=False),
# .env chỉ điền chỗ trống. Không có .env thì im lặng bỏ qua (find_dotenv()
# trả rỗng), không lỗi.
from dotenv import load_dotenv as _load_dotenv
_load_dotenv()

from ..cmcv import Tier
from ..config import JudgeRefineConfig
from ..judge_refine import JudgeRefine, make_judge_fn
from ..sft import HardItem

# ============================================================ SỬA Ở ĐÂY =====
# Mỗi mục: 1 mẫu Hard thật. `image` là đường dẫn ảnh trang (hoặc "file.pdf#0"
# nếu PAGES_ROOT là thư mục PDF). `draft` là bản nháp SAI cần Judge-and-Refine
# sửa — không phải nhãn đúng.
PAGES_ROOT = "data/vietnamese_sample/invoices"
DATASET = [
    {"subtask": "formula", "image": "invoices_0000.jpg",
     "draft": r"E = mc^2"},                     # ví dụ — thay bằng công thức thật trên trang
    {"subtask": "table", "image": "invoices_0001.jpg",
     "draft": "<table><tr><td>Số lượng</td><td>Đơn giá</td></tr></table>"},
    {"subtask": "text", "image": "invoices_0002.jpg",
     "draft": "Hoa don ban hang so 001"},       # thiếu dấu — kịch bản lỗi tiếng Việt thật
]
# =============================================================================


def main() -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print("THIẾU OPENAI_API_KEY. export OPENAI_API_KEY=... rồi chạy lại.")
        return 2

    from ..clients.openai_compat import OpenAICompatClient, OpenAICompatConfig
    from ..clients.cache import DiskCache
    from ..clients.pagestore import PageStore

    root = Path(PAGES_ROOT)
    if not root.exists():
        print(f"Không tìm thấy thư mục ảnh: {root}. Sửa PAGES_ROOT trong file này.")
        return 2

    store = PageStore(root, max_side=1600)
    cache = DiskCache(".cache/ddas_judge_refine_real")
    client = OpenAICompatClient(
        OpenAICompatConfig(model=os.environ.get("JUDGE_MODEL", "gpt-5"),
                           base_url=os.environ.get("OPENAI_BASE_URL",
                                                    "https://api.openai.com/v1"),
                           max_tokens=8192),
        cache)
    judge_fn = make_judge_fn(client.as_call_model("judge-v1"))

    queue = [HardItem(subtask=d["subtask"], page_id=d["image"], tier=Tier.HARD,
                      draft=d["draft"]) for d in DATASET]

    cfg = JudgeRefineConfig()
    print(f"== chạy Judge-and-Refine thật trên {len(queue)} mẫu (model="
          f"{client.cfg.model}, max_rounds={cfg.max_rounds}) ==\n")

    jr = JudgeRefine(judge_fn, store.as_image_fn(), cfg)
    refined, expert = jr.run(queue)

    print(f"-- tự cứu được: {len(refined)}/{len(queue)} --")
    for r in refined:
        print(f"  [{r.subtask}] {r.page_id}  {r.rounds} vòng  conf={r.confidence:.2f}")
        print(f"    -> {r.content[:200]!r}")

    print(f"\n-- chuyển sang người: {len(expert)} --")
    for e in expert:
        print(f"  [{e.subtask}] {e.page_id}  lý do={e.reason}  vòng={e.rounds}")
        if e.note:
            print(f"    ghi chú trọng tài: {e.note[:200]!r}")

    print(f"\n== thống kê ==")
    print(json.dumps({
        "items": jr.stats["items"], "judge_calls": jr.stats["judge_calls"],
        "resolve_rate": round(jr.resolve_rate, 3),
        "mean_rounds": round(jr.mean_rounds, 2),
        "render_failed": jr.stats["render_failed"],
        "openai_calls": client.stats.summary(),
    }, indent=2, ensure_ascii=False))

    print("\nLƯU Ý: 'tự cứu được' đo được resolve_rate, nhưng KHÔNG đo được")
    print("false-clean (trọng tài nhận nhầm là sạch dù vẫn sai) — muốn đo cái")
    print("đó phải biết đáp án đúng của từng mẫu và so `r.content` với nó tay.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
