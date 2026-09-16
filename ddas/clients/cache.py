"""Cache trên đĩa cho mọi lời gọi model — bắt buộc, không phải tối ưu hoá.

Vì sao bắt buộc: pipeline này chạy lại nhiều lần trong quá trình hiệu chuẩn
(đổi tau, đổi chiến lược lấy mẫu, sửa bug ở khâu sau). Mỗi lần chạy lại mà gọi
lại API là trả tiền lần nữa cho ĐÚNG kết quả đã có. Ở quy mô §3.2 (hàng triệu
trang qua Mistral OCR) thì một lần chạy lại vô ý đủ đốt hết ngân sách.

Khoá là NỘI DUNG, không phải page_id: sha256 của (model, phiên bản prompt,
bytes ảnh, tham số sinh). Ba hệ quả có chủ ý:
  - Cùng một trang xuất hiện ở hai pool khác nhau -> chung một ô cache.
  - Đổi prompt -> đổi khoá -> cache cũ tự động không dùng nữa, KHÔNG âm thầm
    trả về kết quả sinh bằng prompt cũ (đây là lỗi im lặng khó tìm nhất).
  - Đổi model/tham số sinh -> cũng đổi khoá.

Ghi bằng tmp-file + rename (atomic trên POSIX) nên chạy đa tiến trình không
sinh file nửa vời; đọc trúng file hỏng thì coi như miss chứ không làm sập job.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
import threading
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from PIL import Image

log = logging.getLogger("ddas.clients.cache")

# Tăng khi format bản ghi cache đổi -> vô hiệu hoá toàn bộ cache cũ.
CACHE_FORMAT_VERSION = "v1"


def image_digest(img: Image.Image) -> str:
    """Hash nội dung ảnh (PNG không nén để nhanh; chỉ cần ổn định, không cần nhỏ)."""
    buf = BytesIO()
    img.save(buf, format="PNG", compress_level=0)
    return hashlib.sha256(buf.getvalue()).hexdigest()


def make_key(model: str, prompt_version: str, *,
             images: Sequence[Image.Image] = (),
             image_digests: Sequence[str] = (),
             texts: Sequence[str] = (),
             params: Optional[Dict[str, Any]] = None) -> str:
    """Khoá cache = sha256 của mọi thứ ảnh hưởng tới output.

    `image_digests` cho phép truyền hash đã tính sẵn (tránh hash lại cùng một
    trang cho 3 model CMCV — ở quy mô triệu trang thì đây là tiền CPU thật).
    """
    h = hashlib.sha256()
    h.update(CACHE_FORMAT_VERSION.encode())
    h.update(b"\x00" + model.encode() + b"\x00" + prompt_version.encode())
    for d in image_digests:
        h.update(b"\x01" + d.encode())
    for img in images:
        h.update(b"\x01" + image_digest(img).encode())
    for t in texts:
        h.update(b"\x02" + t.encode("utf-8"))
    if params:
        h.update(b"\x03" + json.dumps(params, sort_keys=True, ensure_ascii=False).encode())
    return h.hexdigest()


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    writes: int = 0
    corrupt: int = 0


class DiskCache:
    """Kho key-value nén gzip, chia thư mục con theo 2 ký tự đầu của khoá.

    Chia shard vì thư mục phẳng chứa hàng triệu file làm chậm mọi thao tác
    filesystem (và một số fs còn có trần số entry).
    """

    def __init__(self, root: os.PathLike | str, enabled: bool = True):
        self.root = Path(root)
        self.enabled = enabled
        self.stats = CacheStats()
        self._lock = threading.Lock()
        if enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json.gz"

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        p = self._path(key)
        if not p.exists():
            with self._lock:
                self.stats.misses += 1
            return None
        try:
            with gzip.open(p, "rt", encoding="utf-8") as f:
                rec = json.load(f)
        except (OSError, EOFError, json.JSONDecodeError, gzip.BadGzipFile) as e:
            # File hỏng (job bị kill giữa chừng ở bản cũ, đĩa lỗi) -> coi như
            # miss và gọi lại, KHÔNG làm sập cả job vì một ô cache.
            log.warning("cache hỏng %s: %s — coi như miss", p, e)
            with self._lock:
                self.stats.corrupt += 1
                self.stats.misses += 1
            return None
        with self._lock:
            self.stats.hits += 1
        return rec

    def put(self, key: str, value: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
        try:
            with gzip.open(tmp, "wt", encoding="utf-8") as f:
                json.dump(value, f, ensure_ascii=False)
            os.replace(tmp, p)              # atomic: đa tiến trình không thấy file nửa vời
            with self._lock:
                self.stats.writes += 1
        except OSError as e:
            log.warning("không ghi được cache %s: %s", p, e)
            tmp.unlink(missing_ok=True)

    def summary(self) -> Dict[str, Any]:
        s = self.stats
        total = s.hits + s.misses
        return {"hits": s.hits, "misses": s.misses, "writes": s.writes,
                "corrupt": s.corrupt,
                "hit_rate": round(s.hits / total, 4) if total else 0.0,
                "root": str(self.root)}
