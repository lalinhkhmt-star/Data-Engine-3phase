"""Render-then-verify — dựng lại ẢNH từ nhãn có cấu trúc (§3.3 paper, dòng 51-53).

Vì sao phải render thay vì để model tự soi lại output của chính nó: ánh xạ đa
phương thức bất đối xứng. Model mạnh ở chiều ảnh -> chuỗi có cấu trúc, nhưng
yếu ở chiều ngược lại — nó không hình dung được chuỗi LaTeX/HTML sẽ HIỆN RA
thế nào, nên tự phản tỉnh thì gần như luôn kết luận "output đúng rồi". Render
lại thành ảnh rồi đặt cạnh ảnh gốc mới biến lỗi cấu trúc (thiếu dấu căn hàng,
thẻ không đóng, lệch colspan) thành khác biệt NHÌN THẤY ĐƯỢC.

Backend theo thứ tự ưu tiên (cùng kiểu với EmbedConfig.layout_source_order):
  - LaTeX : 'pdflatex' (đúng chuẩn, cần cài TeX) -> 'mathtext' (matplotlib, luôn có)
  - HTML  : 'pymupdf' (pymupdf.Story — đã là dependency sẵn của repo)

CẢNH BÁO quan trọng về backend 'mathtext': nó KHÔNG phải LaTeX đầy đủ. Mọi
environment (\\begin{bmatrix}, \\begin{array}, \\begin{aligned}, \\begin{cases}...)
đều parse lỗi DÙ công thức hoàn toàn hợp lệ — đây là giới hạn của backend, không
phải bằng chứng nhãn sai. Vì vậy `Render` luôn trả kèm `backend`, và
judge_refine.py định tuyến case render lỗi sang hàng đợi người kèm tên backend
để chạy lại được khi có TeX thật, KHÔNG tự kết luận nhãn hỏng.
"""
from __future__ import annotations

import io
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import List, Optional, Sequence

from PIL import Image, ImageOps

LATEX_BACKENDS = ("pdflatex", "mathtext")
RENDERABLE = ("formula", "table")


@dataclass
class Render:
    """Kết quả render. `image is None` <=> thất bại, lý do trong `error`."""
    image: Optional[Image.Image]
    backend: str
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.image is not None


# ------------------------------------------------------------- tiện ích -----

def _trim(img: Image.Image, pad: int = 10) -> Image.Image:
    """Cắt viền trắng thừa — giữ tỉ lệ nội dung lớn trong ảnh để judge nhìn rõ."""
    bbox = ImageOps.invert(img.convert("L")).getbbox()
    if bbox is None:
        return img
    x1, y1, x2, y2 = bbox
    W, H = img.size
    return img.crop((max(x1 - pad, 0), max(y1 - pad, 0),
                     min(x2 + pad, W), min(y2 + pad, H)))


def _vstack(imgs: Sequence[Image.Image]) -> Image.Image:
    """Ghép dọc các trang render (bảng dài tràn nhiều trang)."""
    if len(imgs) == 1:
        return imgs[0]
    W = max(im.width for im in imgs)
    H = sum(im.height for im in imgs)
    out = Image.new("RGB", (W, H), "white")
    y = 0
    for im in imgs:
        out.paste(im, (0, y))
        y += im.height
    return out


def _pdf_to_images(data_or_path, dpi: int) -> List[Image.Image]:
    import pymupdf
    doc = (pymupdf.Document(data_or_path) if isinstance(data_or_path, str)
           else pymupdf.Document("pdf", data_or_path))
    out = []
    for page in doc:
        pix = page.get_pixmap(dpi=dpi)
        out.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
    return out


_MATH_DELIMS = ("$$", "$", r"\[", r"\]", r"\(", r"\)")


def strip_math_delims(s: str) -> str:
    """Bỏ $...$ / \\[...\\] bọc ngoài — model hay trả kèm, render lại thì thừa."""
    s = (s or "").strip()
    for d in _MATH_DELIMS:
        if s.startswith(d):
            s = s[len(d):].strip()
    for d in _MATH_DELIMS:
        if s.endswith(d):
            s = s[: -len(d)].strip()
    return s


# --------------------------------------------------------------- LaTeX ------

_TEX_DOC = r"""\documentclass[preview,border=6pt,12pt]{standalone}
\usepackage{amsmath,amssymb,amsfonts}
\begin{document}
$\displaystyle %s$
\end{document}
"""


def _latex_pdflatex(s: str, dpi: int, timeout: float = 25.0) -> Render:
    exe = shutil.which("pdflatex")
    if exe is None:
        return Render(None, "pdflatex", "pdflatex không có trong PATH")
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "f.tex"), "w", encoding="utf-8") as fh:
            fh.write(_TEX_DOC % s)
        try:
            # -no-shell-escape: nội dung là output model (không tin cậy), tuyệt đối
            # không cho \write18 chạy lệnh hệ thống khi biên dịch.
            p = subprocess.run([exe, "-no-shell-escape", "-interaction=nonstopmode",
                                "-halt-on-error", "f.tex"],
                               cwd=d, capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return Render(None, "pdflatex", f"timeout {timeout}s")
        pdf = os.path.join(d, "f.pdf")
        if p.returncode != 0 or not os.path.exists(pdf):
            log = p.stdout.decode("utf-8", "replace")
            err = next((ln for ln in log.splitlines() if ln.startswith("!")), "biên dịch lỗi")
            return Render(None, "pdflatex", err[:200])
        return Render(_trim(_pdf_to_images(pdf, dpi)[0]), "pdflatex")


def _latex_mathtext(s: str, dpi: int) -> Render:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig = plt.figure(figsize=(0.01, 0.01))
    buf = io.BytesIO()
    try:
        fig.text(0, 0, f"${s}$", fontsize=20)
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                    pad_inches=0.1, facecolor="white")
    except Exception as e:                       # mathtext ném ValueError khi parse hỏng
        return Render(None, "mathtext", f"{type(e).__name__}: {str(e).strip().splitlines()[-1][:160]}")
    finally:
        plt.close(fig)
    buf.seek(0)
    return Render(_trim(Image.open(buf).convert("RGB")), "mathtext")


def available_latex_backends(backends: Sequence[str] = LATEX_BACKENDS) -> List[str]:
    """Lọc bỏ backend chưa cài — để lý do lỗi báo về luôn là lỗi NỘI DUNG của
    backend đã chạy thật, không phải "chưa cài TeX" (hai việc khác hẳn nhau khi
    quyết định có đẩy mẫu sang người hay không)."""
    return [b for b in backends if b != "pdflatex" or shutil.which("pdflatex")]


def render_latex(latex: str, dpi: int = 150,
                 backends: Sequence[str] = LATEX_BACKENDS) -> Render:
    """Compile công thức LaTeX thành ảnh, thử lần lượt theo `backends`.

    Trả Render của backend đầu tiên thành công; nếu tất cả backend KHẢ DỤNG đều
    hỏng thì trả lỗi của backend ưu tiên cao nhất trong số đã chạy thật.
    """
    s = strip_math_delims(latex)
    avail = available_latex_backends(backends)
    if not avail:
        return Render(None, "none", "không backend LaTeX nào khả dụng")
    if not s:
        return Render(None, avail[0], "nội dung rỗng")
    first: Optional[Render] = None
    for b in avail:
        r = _latex_pdflatex(s, dpi) if b == "pdflatex" else _latex_mathtext(s, dpi)
        if r.ok:
            return r
        first = first or r
    return first


# ---------------------------------------------------------------- HTML ------

_TABLE_CSS = ("table{border-collapse:collapse;font-size:11px}"
              "td,th{border:1px solid #333;padding:3px 6px}")


def render_table_html(html: str, dpi: int = 150, width: float = 1000.0,
                      height: float = 1400.0, max_pages: int = 4) -> Render:
    """Render bảng HTML thành ảnh bằng pymupdf.Story.

    Bảng dài tràn nhiều trang được ghép dọc lại thành MỘT ảnh — cắt bớt sẽ làm
    judge kết luận "thiếu hàng" trong khi nhãn không hề thiếu.
    """
    if not (html or "").strip():
        return Render(None, "pymupdf", "nội dung rỗng")
    try:
        import pymupdf
        buf = io.BytesIO()
        writer = pymupdf.DocumentWriter(buf)
        story = pymupdf.Story(html, user_css=_TABLE_CSS)
        mb = pymupdf.Rect(0, 0, width, height)
        where = mb + (20, 20, -20, -20)
        more, n = 1, 0
        while more and n < max_pages:
            dev = writer.begin_page(mb)
            more, _ = story.place(where)
            story.draw(dev)
            writer.end_page()
            n += 1
        writer.close()
        pages = [_trim(im) for im in _pdf_to_images(buf.getvalue(), dpi)]
    except Exception as e:
        return Render(None, "pymupdf", f"{type(e).__name__}: {str(e)[:160]}")
    if not pages:
        return Render(None, "pymupdf", "không sinh được trang nào")
    return Render(_vstack(pages), "pymupdf")


# ------------------------------------------------------------ dispatch ------

def render_label(content: str, subtask: str, dpi: int = 150,
                 latex_backends: Sequence[str] = LATEX_BACKENDS) -> Optional[Render]:
    """Render nhãn theo subtask. None = subtask không có đường render.

    Chỉ formula (LaTeX) và table (HTML) render được — đúng phạm vi paper nêu.
    text/layout không có dạng biểu diễn để dựng lại ảnh, judge so trực tiếp với
    ảnh gốc (xem judge_refine.py).
    """
    if subtask == "formula":
        return render_latex(content, dpi, latex_backends)
    if subtask == "table":
        return render_table_html(content, dpi)
    return None
