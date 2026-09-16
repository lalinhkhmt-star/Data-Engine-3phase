"""Nguồn ảnh trang — hiện thực `image_fn: Callable[[str], Image.Image]`.

Pool mục tiêu là 100% tài liệu SCAN, nên ẢNH là đầu vào gốc chứ không phải
text layer: không có nhánh PyMuPDF trích chữ ở đây (xem layout_prior.py để
biết vì sao nhánh đó chỉ còn ý nghĩa với born-digital).

Một page_id định danh duy nhất một trang, theo hai dạng:
  "ten/file.png"        -> ảnh rời trong `root`
  "ten/file.pdf#7"      -> trang thứ 7 (đánh số từ 0) của PDF scan, render `dpi`

Ba thứ module này chịu trách nhiệm, đều là tiền thật ở quy mô triệu trang:

  1. LRU trong RAM. CMCV gọi image_fn tối đa 3 lần cho CÙNG một trang (target/
     cheap/expensive), §3.3 gọi lại một lần nữa cho mỗi vòng judge. Đọc đĩa +
     decode JPEG 4 lần cho một trang là lãng phí thuần tuý.
  2. CHUẨN HOÁ KÍCH THƯỚC trước khi gửi model. Ảnh scan thật hay là 300dpi A4
     (~2480x3508). Gửi nguyên cỡ đó cho VLM vừa vượt giới hạn ảnh của nhà cung
     cấp vừa tốn token ảnh vô ích. Hạ cạnh dài về `max_side` (mặc định 1600px,
     tương đương ~200dpi A4 — đúng cỡ costmodel.py dùng để ước token).
  3. Ổn định hash. Cùng một page_id phải cho ra ĐÚNG cùng một ảnh giữa các lần
     chạy, nếu không khoá cache đổi và cache mất tác dụng hoàn toàn. Nên mọi
     phép resize đều tất định (LANCZOS, cùng công thức làm tròn) và ảnh được
     ép về RGB.
"""
from __future__ import annotations

import logging
import os
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

from PIL import Image

log = logging.getLogger("ddas.clients.pagestore")

IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"})


class PageNotFound(FileNotFoundError):
    pass


def _resize_long_side(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    m = max(w, h)
    if max_side <= 0 or m <= max_side:
        return img
    s = max_side / m
    # round() chứ không int(): tất định và không lệch 1px giữa các phiên bản Pillow
    return img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)


class PageStore:
    """Tra ảnh theo page_id, có LRU. `as_image_fn()` trả đúng callable core cần."""

    def __init__(self, root: os.PathLike | str, *, dpi: int = 200,
                 max_side: int = 1600, lru_size: int = 64,
                 index: Optional[Dict[str, str]] = None):
        self.root = Path(root)
        self.dpi = dpi
        self.max_side = max_side
        self.lru_size = max(1, lru_size)
        # index tuỳ chọn: page_id -> đường dẫn tương đối, dùng khi page_id là mã
        # sinh ra (hash, id trong DB) chứ không phải tên file.
        self.index = index or {}
        self._cache: "OrderedDict[str, Image.Image]" = OrderedDict()
        self._lock = threading.Lock()
        self.stats = {"loads": 0, "lru_hits": 0, "pdf_renders": 0}

    # ------------------------------------------------------------ tra cứu --
    def _resolve(self, page_id: str) -> Tuple[Path, Optional[int]]:
        ref = self.index.get(page_id, page_id)
        page_no: Optional[int] = None
        if "#" in ref:
            ref, _, frag = ref.rpartition("#")
            try:
                page_no = int(frag)
            except ValueError as e:
                raise PageNotFound(f"page_id {page_id!r}: '#{frag}' không phải số trang") from e
        p = Path(ref)
        if not p.is_absolute():
            p = self.root / p
        if not p.exists():
            raise PageNotFound(f"page_id {page_id!r} -> không có file {p}")
        return p, page_no

    def _load(self, page_id: str) -> Image.Image:
        path, page_no = self._resolve(page_id)
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            if page_no is None:
                raise PageNotFound(f"page_id {page_id!r} trỏ vào PDF nhưng thiếu '#số_trang'")
            img = self._render_pdf_page(path, page_no)
            self.stats["pdf_renders"] += 1
        elif suffix in IMAGE_SUFFIXES:
            img = Image.open(path)
            img.load()
        else:
            raise PageNotFound(f"page_id {page_id!r}: đuôi {suffix!r} không được hỗ trợ")
        self.stats["loads"] += 1
        return _resize_long_side(img.convert("RGB"), self.max_side)

    def _render_pdf_page(self, path: Path, page_no: int) -> Image.Image:
        import pymupdf                      # nạp muộn: chỉ cần khi pool có PDF

        with pymupdf.open(path) as doc:
            if not 0 <= page_no < doc.page_count:
                raise PageNotFound(f"{path}: không có trang {page_no} (tổng {doc.page_count})")
            pix = doc[page_no].get_pixmap(dpi=self.dpi)
            return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

    # -------------------------------------------------------------- public --
    def get(self, page_id: str) -> Image.Image:
        with self._lock:
            img = self._cache.get(page_id)
            if img is not None:
                self._cache.move_to_end(page_id)
                self.stats["lru_hits"] += 1
                return img
        img = self._load(page_id)           # nạp NGOÀI lock: I/O chậm, đừng chặn thread khác
        with self._lock:
            self._cache[page_id] = img
            while len(self._cache) > self.lru_size:
                self._cache.popitem(last=False)
        return img

    def as_image_fn(self) -> Callable[[str], Image.Image]:
        return self.get

    def discover(self, pattern: str = "**/*") -> list[str]:
        """Liệt kê page_id có trong `root` — dùng cho pilot/dev-set, không dùng ở
        quy mô pool thật (ở đó danh sách trang đến từ manifest của bước ingest)."""
        out: list[str] = []
        for p in sorted(self.root.glob(pattern)):
            if not p.is_file():
                continue
            rel = p.relative_to(self.root).as_posix()
            if p.suffix.lower() in IMAGE_SUFFIXES:
                out.append(rel)
            elif p.suffix.lower() == ".pdf":
                import pymupdf
                with pymupdf.open(p) as doc:
                    out.extend(f"{rel}#{i}" for i in range(doc.page_count))
        return out
