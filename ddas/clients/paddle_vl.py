"""Runner CMCV cho PaddleOCR-VL (EXPENSIVE_EXTERNAL) — TỰ HOST, chỉ tốn GPU-giờ.

Tên biến "expensive" kế thừa từ bản paper gốc (nơi model thứ ba là API đắt). Ở
repo này nó chỉ còn nghĩa "gọi hiếm vì cascade cắt được", KHÔNG còn nghĩa đắt
tiền: PaddleOCR-VL chỉ 0.9B tham số, đo được 1.224 trang/s trên A100 — rẻ hơn
cả target 8B (xem costmodel.py).

SCHEMA PHỤ THUỘC CÁCH BẠN DEPLOY — đây là điểm phải kiểm chứng đầu tiên khi có
endpoint thật. Ba đường deploy phổ biến trả ba shape khác nhau:
  - PaddleX serving `/layout-parsing`  -> {"result": {"layoutParsingResults": [...]}}
  - FastDeploy / vLLM-style            -> giao thức OpenAI, dùng qwen_vl.py thay vì file này
  - server tự viết                     -> tuỳ bạn
Nên adapter nhận `extract_fn` để cắm hàm bóc riêng. Mặc định dò theo PaddleX.
Đừng tin mặc định này cho tới khi gọi thử một trang và nhìn tận mắt response —
sai shape sẽ ra ParseResult rỗng, mà ParseResult rỗng thì "đồng thuận" với mọi
ParseResult rỗng khác và sinh nhãn Easy rỗng (xem cảnh báo trong qwen_vl.py).
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional

from PIL import Image

from ..cmcv import ParseResult
from .base import ClientError, ClientStats, HttpClient
from .cache import DiskCache, image_digest, make_key
from .normalize import blocks_to_parse_result
from .openai_compat import encode_image

log = logging.getLogger("ddas.clients.paddle")

PADDLE_PROMPT_VERSION = "paddleocr-vl-v1"

# extract_fn(response_json) -> list block dạng {bbox, type, content}
ExtractFn = Callable[[Dict[str, Any]], List[Dict[str, Any]]]


def paddlex_extract(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Bóc block từ response PaddleX `/layout-parsing` (shape mặc định)."""
    res = data.get("result") or data
    items = (res.get("layoutParsingResults") or res.get("layout_parsing_results")
             or res.get("parsingResList") or [])
    if not items:
        return []
    first = items[0] if isinstance(items, list) else items
    blocks = (first.get("prunedResult") or first.get("pruned_result") or first)
    if isinstance(blocks, dict):
        blocks = (blocks.get("parsing_res_list") or blocks.get("parsingResList")
                  or blocks.get("boxes") or [])
    out: List[Dict[str, Any]] = []
    for b in blocks or []:
        if not isinstance(b, dict):
            continue
        out.append({
            "bbox": b.get("block_bbox") or b.get("bbox") or b.get("coordinate"),
            "type": b.get("block_label") or b.get("label") or b.get("type"),
            "content": b.get("block_content") or b.get("content") or b.get("text") or "",
        })
    return out


class PaddleVLRunner:
    def __init__(self, url: str, image_fn: Callable[[str], Image.Image], *,
                 model_name: str = "paddleocr-vl",
                 extract_fn: ExtractFn = paddlex_extract,
                 payload_fn: Optional[Callable[[str], Dict[str, Any]]] = None,
                 box_space: str = "abs",
                 cache: Optional[DiskCache] = None,
                 rps: float = 0.0,
                 timeout: float = 240.0,
                 stats: Optional[ClientStats] = None):
        self.url = url
        self.image_fn = image_fn
        self.model_name = model_name     # phải khớp cmcv.EXPENSIVE_EXTERNAL
        self.extract_fn = extract_fn
        self.payload_fn = payload_fn or (lambda b64: {"file": b64, "fileType": 1})
        self.box_space = box_space
        self.cache = cache
        self.http = HttpClient(model_name, timeout=timeout, rps=rps, stats=stats)
        self.stats_local = {"pages": 0, "empty": 0}

    def __call__(self, page_id: str) -> ParseResult:
        img = self.image_fn(page_id)
        ck = make_key(self.model_name, PADDLE_PROMPT_VERSION,
                      image_digests=[image_digest(img)], params={"url": self.url})
        data = self.cache.get(ck) if self.cache is not None else None
        if data is not None:
            self.http.stats.add(cache_hits=1)
        else:
            t0 = time.monotonic()
            data = self.http.post_json(self.url, self.payload_fn(encode_image(img)),
                                       {"Content-Type": "application/json"})
            data["_latency_ms"] = (time.monotonic() - t0) * 1000.0
            if self.cache is not None:
                self.cache.put(ck, data)

        blocks = self.extract_fn(data)
        if not blocks:
            # KHÔNG trả ParseResult rỗng: xem docstring đầu file. Trang trắng
            # thật cũng đi qua nhánh này, và đó là cái giá chấp nhận được —
            # cách ly nhầm một trang trắng rẻ hơn nhiều so với bơm nhãn rỗng
            # vào tập train dưới mác Easy.
            self.stats_local["empty"] += 1
            raise ClientError(
                f"[{self.model_name}] {page_id}: extract_fn không bóc được block nào. "
                "Kiểm tra shape response có khớp extract_fn không (xem paddlex_extract).")

        self.stats_local["pages"] += 1
        W, H = img.size
        return blocks_to_parse_result(page_id, self.model_name, blocks, W, H,
                                      box_space=self.box_space,
                                      latency_ms=float(data.get("_latency_ms") or 0.0))
