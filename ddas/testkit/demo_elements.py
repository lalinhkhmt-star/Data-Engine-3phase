"""Kiểm Stage 2 (mức element) — chặn lỗi "đồng thuận rỗng" ở mức element.

    python3 -m ddas.testkit.demo_elements

Bài test này ra đời sau khi phát hiện subtask `text` — subtask có ngân sách
lớn nhất (25M/60M = 42% dataset) — sinh ra 100% bản ghi SFT có NHÃN RỖNG gán
tier EASY. Nguyên nhân: `ParseResult` giữ text là một chuỗi ĐàGHÉP nên
`_elements_of` không lấy lại được nội dung từng block và gán "" cho mọi element
text; hai chuỗi rỗng thì "đồng thuận" tuyệt đối => EASY.

Đây là lần THỨ HAI cùng một loại lỗi lọt qua (lần đầu ở mức trang, xem
demo_clients.py bài [6b]), nên bộ test dưới đây phủ cả các biến thể:
rỗng-cả-hai, rỗng-một-bên, và trường hợp adapter cũ không có `contents`.
"""
from __future__ import annotations

import sys
from typing import List, Optional

import numpy as np

from ..cmcv import CHEAP_EXTERNAL, TARGET_MODEL, ParseResult, Tier
from ..element import SUBTASK_OF_LABEL, derive_element_cmcv
from ..layout_heron import LayoutBox
from ..sft import assemble_element_records

_results: List[bool] = []


def check(cond: bool, msg: str) -> bool:
    _results.append(bool(cond))
    print(f"  {'✓' if cond else '✗'} {msg}")
    return bool(cond)


BOXES = {
    "text":    np.array([10., 10., 300., 60.], np.float32),
    "title":   np.array([10., 80., 300., 130.], np.float32),
    "list":    np.array([10., 150., 300., 200.], np.float32),
    "caption": np.array([10., 220., 300., 260.], np.float32),
    "table":   np.array([10., 280., 300., 400.], np.float32),
    "formula": np.array([10., 420., 300., 470.], np.float32),
}
CLS = {"text": "Text", "title": "Title", "list": "List-item",
       "caption": "Caption", "table": "Table", "formula": "Formula"}


def anchors(labels: List[str]) -> List[LayoutBox]:
    return [LayoutBox(box=BOXES[l], cls=CLS[l], score=0.99) for l in labels]


def result(model: str, labels: List[str], contents: List[Optional[str]]) -> ParseResult:
    return ParseResult(
        page_id="p", model=model,
        boxes=np.stack([BOXES[l] for l in labels]), labels=list(labels),
        contents=["" if c is None else c for c in contents],
        text="\n".join(c for c in contents if c),
        tables=[c for l, c in zip(labels, contents) if l == "table" and c],
        formulas=[c for l, c in zip(labels, contents) if l == "formula" and c])


def main() -> int:
    labels = ["text", "title", "list", "caption", "table", "formula"]
    good = ["Doanh thu thuần quý III tăng 12% so với cùng kỳ.",
            "BÁO CÁO TÀI CHÍNH QUÝ III",
            "Chi phí bán hàng giảm 3%",
            "Bảng 1: Kết quả kinh doanh",
            "<table><tr><td>Doanh thu</td><td>1250</td></tr></table>",
            "E = mc^2"]

    print("\n[1] Element text/title/list/caption có NỘI DUNG THẬT và vào subtask 'text'")
    els = derive_element_cmcv("p", anchors(labels),
                              result(TARGET_MODEL, labels, good),
                              result(CHEAP_EXTERNAL, labels, good), None)
    check(len(els) == 6, f"sinh đủ {len(els)}/6 element")
    by_lbl = {l: e for l, e in zip(labels, els)}
    for l in ("text", "title", "list", "caption"):
        e = by_lbl[l]
        check(e.etype == "text" and e.content.get(TARGET_MODEL),
              f"{l:8s} -> subtask={e.etype:7s} tier={e.tier.value:6s} "
              f"content={e.content.get(TARGET_MODEL, '')[:34]!r}")
    check(by_lbl["table"].etype == "table" and by_lbl["formula"].etype == "formula",
          "table/formula giữ nguyên subtask riêng")
    check(all(e.tier == Tier.EASY for e in els),
          "2 model khớp nội dung thật => EASY (nhãn có nội dung)")

    print("\n[2] Bản ghi SFT không còn nhãn rỗng")
    by_st = {}
    for e in els:
        by_st.setdefault(e.etype, []).append(e)
    ready, hard = assemble_element_records(by_st)
    empty = [r for r in ready if not r.content]
    check(not empty, f"{len(empty)}/{len(ready)} bản ghi SFT có nhãn rỗng (phải = 0)")
    n_text = sum(1 for r in ready if r.subtask == "text")
    check(n_text == 4, f"subtask 'text' có {n_text} bản ghi (text+title+list+caption)")

    print("\n[3] CẢ HAI model cùng rỗng => HARD, KHÔNG phải EASY rỗng")
    els = derive_element_cmcv("p", anchors(labels),
                              result(TARGET_MODEL, labels, [""] * 6),
                              result(CHEAP_EXTERNAL, labels, [""] * 6), None)
    check(all(e.tier == Tier.HARD for e in els),
          f"tier: {sorted({e.tier.value for e in els})}")
    ready, hard = assemble_element_records({"text": [e for e in els if e.etype == "text"]})
    check(not ready, f"0 bản ghi SFT sinh ra từ element rỗng (thực tế {len(ready)})")

    print("\n[4] MỘT bên rỗng, một bên có chữ => bất đồng thật, không EASY")
    els = derive_element_cmcv("p", anchors(["text"]),
                              result(TARGET_MODEL, ["text"], [""]),
                              result(CHEAP_EXTERNAL, ["text"], [good[0]]), None)
    check(els[0].tier != Tier.EASY,
          f"tier = {els[0].tier.value} (sim thấp tự đẩy xuống, không bị chặn oan)")

    print("\n[5] Adapter cũ KHÔNG có `contents` => text là 'không biết', không phải rỗng")
    old_m = ParseResult(page_id="p", model=TARGET_MODEL,
                        boxes=np.stack([BOXES["text"]]), labels=["text"],
                        text="Doanh thu thuần quý III tăng 12%.")   # contents rỗng
    old_p = ParseResult(page_id="p", model=CHEAP_EXTERNAL,
                        boxes=np.stack([BOXES["text"]]), labels=["text"],
                        text="Doanh thu thuần quý III tăng 12%.")
    els = derive_element_cmcv("p", anchors(["text"]), old_m, old_p, None)
    check(els[0].tier == Tier.HARD,
          f"tier = {els[0].tier.value} (thiếu bằng chứng, KHÔNG suy ra EASY rỗng)")

    print("\n[6] Bảng/công thức vẫn hoạt động khi chỉ có formulas/tables (không contents)")
    old_m = ParseResult(page_id="p", model=TARGET_MODEL,
                        boxes=np.stack([BOXES["table"], BOXES["formula"]]),
                        labels=["table", "formula"],
                        tables=[good[4]], formulas=[good[5]])
    old_p = ParseResult(page_id="p", model=CHEAP_EXTERNAL,
                        boxes=np.stack([BOXES["table"], BOXES["formula"]]),
                        labels=["table", "formula"],
                        tables=[good[4]], formulas=[good[5]])
    els = derive_element_cmcv("p", anchors(["table", "formula"]), old_m, old_p, None)
    check(all(e.tier == Tier.EASY for e in els) and all(e.content for e in els),
          f"table/formula vẫn EASY có nội dung: {[e.tier.value for e in els]}")

    print("\n[7] assign_tier: thiếu model thứ 3 KHÔNG mặc nhiên là EASY")
    from ..cmcv import assign_tier
    check(assign_tier(0.99, None, None, 0.92) == Tier.EASY,
          "s_mp cao + cascade cắt  -> EASY (đường cascade hợp lệ)")
    check(assign_tier(0.20, None, None, 0.92) == Tier.HARD,
          "s_mp THẤP + thiếu model 3 -> HARD (trước đây trả EASY kèm nhãn sai)")
    check(assign_tier(0.99, None, None, 0.92, require_3way=True) == Tier.HARD,
          "require_3way mà chỉ có 2 ý kiến -> HARD")

    print("\n[8] SUBTASK_OF_LABEL phủ đúng taxonomy chuẩn")
    from ..clients.normalize import CANON_LABELS
    check(set(CANON_LABELS) == set(SUBTASK_OF_LABEL),
          f"khớp CANON_LABELS: {sorted(SUBTASK_OF_LABEL)}")

    n_ok = sum(_results)
    print(f"\n{'='*64}\n{n_ok}/{len(_results)} kiểm tra ĐẠT")
    return 0 if n_ok == len(_results) else 1


if __name__ == "__main__":
    sys.exit(main())
