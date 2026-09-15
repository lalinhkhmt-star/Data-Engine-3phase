"""Chạy thử §3.3 Judge-and-Refine end-to-end bằng model trọng tài GIẢ LẬP.

Mục đích: kiểm tra đường đi của dữ liệu và cả 6 lối thoát của vòng lặp — KHÔNG
phải đo chất lượng sửa nhãn (điều đó cần model thật, xem IMPLEMENTATION_STATUS.md).
Phần render thì chạy THẬT (matplotlib mathtext / pymupdf.Story).

    python3 -m ddas.testkit.demo_judge_refine
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from ddas.cmcv import Tier
from ddas.config import JudgeRefineConfig
from ddas.judge_refine import (JudgeRefine, JudgeVerdict, prioritize, summarize,
                               weakness_by_subtask)
from ddas.cmcv import CMCVRecord
from ddas.sft import HardItem

# Mỗi mẫu dựng để rơi vào đúng MỘT lối thoát của vòng lặp.
QUEUE = [
    HardItem("formula", "p1", Tier.HARD, draft=r"\frac{a}{b} + \sum_{i=1}^{n} x_i"),   # sửa 1 vòng là xong
    HardItem("formula", "p2", Tier.HARD, draft=r"\frac{a}{"),                          # render lỗi
    HardItem("formula", "p3", Tier.HARD, draft=r"\alpha + \beta"),                     # sửa mãi không đổi -> stuck
    HardItem("table", "p4", Tier.HARD, draft="<table><tr><td>a</td><td>b</td></tr></table>"),   # hết vòng
    HardItem("table", "p5", Tier.HARD, draft="<table><tr><td>x</td></tr></table>"),    # khoanh được lỗi, không sửa nổi
    HardItem("layout", "p6", Tier.HARD),                                               # không có nháp
]


class FakeJudge:
    """Trọng tài giả lập — kịch bản cố định theo page_id, không gọi model nào."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, orig: Image.Image, shot, content: str, subtask: str) -> JudgeVerdict:
        self.calls += 1
        if "p1" in content or content.startswith(r"\frac{a}{b}"):
            return JudgeVerdict(has_error=False, confidence=0.93)
        if content.startswith(r"\alpha"):
            # "sửa" nhưng chỉ đổi khoảng trắng -> similarity ~1 -> stuck
            return JudgeVerdict(True, 0.88, content + " ", "thiếu chỉ số dưới")
        if "<td>x</td>" in content:
            return JudgeVerdict(True, 0.81, None, "cột bị gộp nhầm, không dựng lại được")
        # bảng p4: mỗi vòng sửa một ít nhưng không bao giờ sạch lỗi
        return JudgeVerdict(True, 0.55, content.replace("</table>",
                            f"<tr><td>r{self.calls}</td><td>-</td></tr></table>"),
                            "thiếu hàng")


def fake_page_image(page_id: str) -> Image.Image:
    return Image.new("RGB", (800, 1000), "white")


def main() -> None:
    judge = FakeJudge()
    cfg = JudgeRefineConfig(max_rounds=3)
    jr = JudgeRefine(judge, fake_page_image, cfg)
    refined, expert = jr.run(QUEUE)

    print(f"hàng đợi Hard: {len(QUEUE)} mẫu · lời gọi trọng tài: {jr.stats['judge_calls']} · "
          f"tự cứu {100*jr.resolve_rate:.0f}% · {jr.mean_rounds:.1f} vòng/mẫu")

    print("\n-- tự sửa được --")
    for r in refined:
        print(f"  {r.subtask:8} {r.page_id}  {r.rounds} vòng  conf={r.confidence:.2f}  "
              f"-> SFT(label_source={r.to_sft().label_source})")

    # Ưu tiên #2 cần độ yếu theo subtask, lấy từ sims mà §3.2 đã tính sẵn.
    recs = [CMCVRecord("p1", {}, {"formula": {"M-P": 0.42, "M-Q": 0.38},
                                  "table": {"M-P": 0.71, "M-Q": 0.66},
                                  "layout": {"M-P": 0.80, "M-Q": None}}, True, {})]
    w = weakness_by_subtask(recs)
    print("\n-- độ yếu theo subtask (1 - đồng thuận target/external) --")
    print("  " + "  ".join(f"{k}={v:.2f}" for k, v in sorted(w.items())))

    print("\n-- hàng đợi người, đã xếp ưu tiên --")
    for i, e in enumerate(prioritize(expert, w, cfg.min_confidence), 1):
        tag = "khoanh được lỗi" if e.located and e.confidence >= cfg.min_confidence else "làm lại từ đầu"
        print(f"  {i}. {e.subtask:8} {e.page_id}  {e.reason:14} conf={e.confidence:.2f}  "
              f"[{tag}]  {e.note or e.backend}")

    print("\n-- thống kê theo subtask --")
    for st, s in summarize(refined, expert).items():
        print(f"  {st:8} refined={s['refined']} expert={s['expert']} "
              f"tự cứu={100*s['resolve_rate']:.0f}% lý do={s['lý do']}")


if __name__ == "__main__":
    main()
