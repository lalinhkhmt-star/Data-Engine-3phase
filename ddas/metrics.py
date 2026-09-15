"""Độ đo đồng thuận theo từng subtask, dùng cho CMCV.

Mỗi hàm trả về similarity trong [0, 1]; càng cao càng đồng thuận.
Bản proxy ở đây chạy thuần CPU/O(n) để scale tới hàng chục triệu trang.
Bản chính thức (apted-TEDS, CDM render-based) cắm qua cùng chữ ký hàm.
"""
from __future__ import annotations

import re
import unicodedata
from html.parser import HTMLParser
from typing import List, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------- text ------

_WS = re.compile(r"\s+")


def norm_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = _WS.sub(" ", s).strip()
    return s


def _lev(a: Sequence, b: Sequence, cap: int | None = None) -> int:
    """Levenshtein O(len(a)*len(b)) bộ nhớ O(min)."""
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
        if cap is not None and min(prev) > cap:
            return cap + 1
    return prev[-1]


def ned(a: str, b: str) -> float:
    """Normalized edit distance -> similarity 1-NED."""
    a, b = norm_text(a), norm_text(b)
    if not a and not b:
        return 1.0
    n = max(len(a), len(b))
    return 1.0 - _lev(a, b) / n


def text_sim(a: str, b: str) -> float:
    return ned(a, b)


# ------------------------------------------------------------- formula ------

_TEX_STRIP = re.compile(r"\\(?:left|right|,|;|!|quad|qquad|displaystyle|limits)\b|\s+|{|}")
_TEX_ALIAS = {r"\dfrac": r"\frac", r"\tfrac": r"\frac", r"\ast": "*", r"\cdot": r"\cdot"}


def canon_latex(s: str) -> List[str]:
    s = unicodedata.normalize("NFKC", s or "")
    for k, v in _TEX_ALIAS.items():
        s = s.replace(k, v)
    toks = re.findall(r"\\[a-zA-Z]+|\d|[^\s]", _TEX_STRIP.sub(" ", s))
    return [t for t in toks if t.strip()]


def cdm_proxy(a: str, b: str) -> float:
    """Xấp xỉ CDM bằng edit distance trên chuỗi token LaTeX đã chuẩn hoá.

    Bất biến với whitespace / \\left\\right / {} thừa — tức là chỉ phạt khác biệt
    ngữ nghĩa, giống tinh thần của CDM (so khớp ký tự sau khi render).
    """
    ta, tb = canon_latex(a), canon_latex(b)
    if not ta and not tb:
        return 1.0
    n = max(len(ta), len(tb))
    return 1.0 - _lev(ta, tb) / n


formula_sim = cdm_proxy


# --------------------------------------------------------------- table ------

class _TableParser(HTMLParser):
    """Trích cấu trúc lưới + nội dung ô từ HTML table."""

    def __init__(self) -> None:
        super().__init__()
        self.rows: List[List[Tuple[int, int, str]]] = []
        self._cur: List[Tuple[int, int, str]] | None = None
        self._buf: List[str] = []
        self._span = (1, 1)
        self._in_cell = False

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self._cur = []
        elif tag in ("td", "th"):
            d = dict(attrs)
            self._span = (int(d.get("rowspan", 1) or 1), int(d.get("colspan", 1) or 1))
            self._buf, self._in_cell = [], True

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._in_cell:
            if self._cur is None:
                self._cur = []
            self._cur.append((*self._span, norm_text("".join(self._buf))))
            self._in_cell = False
        elif tag == "tr" and self._cur is not None:
            self.rows.append(self._cur)
            self._cur = None

    def handle_data(self, data):
        if self._in_cell:
            self._buf.append(data)

    def close(self):
        super().close()
        if self._cur:
            self.rows.append(self._cur)


def parse_table(html: str):
    p = _TableParser()
    try:
        p.feed(html or "")
        p.close()
    except Exception:
        pass
    return p.rows


def _struct_seq(rows) -> List[str]:
    out: List[str] = []
    for r in rows:
        out.append("<tr>")
        out.extend(f"c{rs}x{cs}" for rs, cs, _ in r)
    return out


def teds_struct(a_html: str, b_html: str) -> float:
    """TEDS-Struct proxy: edit distance trên chuỗi cấu trúc (bỏ nội dung)."""
    sa, sb = _struct_seq(parse_table(a_html)), _struct_seq(parse_table(b_html))
    if not sa and not sb:
        return 1.0
    n = max(len(sa), len(sb))
    return 1.0 - _lev(sa, sb) / n


def teds(a_html: str, b_html: str, w_struct: float = 0.5) -> float:
    """TEDS proxy = w*cấu trúc + (1-w)*nội dung ô sau khi căn theo vị trí lưới."""
    ra, rb = parse_table(a_html), parse_table(b_html)
    s = teds_struct(a_html, b_html)
    ca = [c for r in ra for (_rs, _cs, c) in r]
    cb = [c for r in rb for (_rs, _cs, c) in r]
    if not ca and not cb:
        return s
    m = min(len(ca), len(cb))
    n = max(len(ca), len(cb))
    content = sum(ned(x, y) for x, y in zip(ca[:m], cb[:m])) / n if n else 1.0
    return w_struct * s + (1.0 - w_struct) * content


table_sim = teds


# -------------------------------------------------------------- layout ------

def iou_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """A:(n,4) B:(m,4) dạng xyxy -> IoU (n,m), vector hoá."""
    if len(A) == 0 or len(B) == 0:
        return np.zeros((len(A), len(B)), dtype=np.float32)
    A = A.astype(np.float32)[:, None, :]
    B = B.astype(np.float32)[None, :, :]
    x1 = np.maximum(A[..., 0], B[..., 0])
    y1 = np.maximum(A[..., 1], B[..., 1])
    x2 = np.minimum(A[..., 2], B[..., 2])
    y2 = np.minimum(A[..., 3], B[..., 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aA = (A[..., 2] - A[..., 0]) * (A[..., 3] - A[..., 1])
    aB = (B[..., 2] - B[..., 0]) * (B[..., 3] - B[..., 1])
    return inter / np.clip(aA + aB - inter, 1e-6, None)


def layout_sim(boxes_a: np.ndarray, labels_a: Sequence[str],
               boxes_b: np.ndarray, labels_b: Sequence[str],
               iou_thr: float = 0.5) -> float:
    """F1 của phép ghép greedy có ràng buộc cùng nhãn."""
    na, nb = len(boxes_a), len(boxes_b)
    if na == 0 and nb == 0:
        return 1.0
    if na == 0 or nb == 0:
        return 0.0
    M = iou_matrix(np.asarray(boxes_a), np.asarray(boxes_b))
    same = np.array([[la == lb for lb in labels_b] for la in labels_a])
    M = np.where(same, M, 0.0)
    used_a, used_b, tp = set(), set(), 0
    order = np.argsort(M, axis=None)[::-1]
    for flat in order:
        i, j = divmod(int(flat), nb)
        if M[i, j] < iou_thr:
            break
        if i in used_a or j in used_b:
            continue
        used_a.add(i); used_b.add(j); tp += 1
    prec, rec = tp / na, tp / nb
    return 0.0 if tp == 0 else 2 * prec * rec / (prec + rec)


SIM_FN = {"text": text_sim, "formula": formula_sim, "table": table_sim}
