"""Runner CMCV cho Mistral OCR (CHEAP_EXTERNAL) — qua API, tính tiền theo trang.

Đây là model chạy trên TOÀN BỘ candidate set (không được cascade cắt, vì cascade
chỉ cắt model thứ ba — xem cmcv.py), nên nó là dòng chi phí API lớn nhất của
§3.2: ~$4/1000 trang đồng bộ, $2/1000 qua Batch API. Ở quy mô 60M trang đó là
$240K/$120K. Hai hệ quả thực tế:

  - LUÔN bật cache (clients/cache.py). Chạy lại pipeline mà thiếu cache là trả
    tiền lần hai cho cùng một kết quả.
  - Cân nhắc Batch API cho các đợt lớn (-50%). Chưa hiện thực ở đây vì batch là
    luồng bất đồng bộ (nộp job -> chờ -> lấy kết quả), không khớp chữ ký
    `page_id -> ParseResult` của CMCV; muốn dùng thì phải chạy một pass gom
    trước rồi đổ vào cache, sau đó CMCV chạy hoàn toàn bằng cache-hit.

CẢNH BÁO VỀ SCHEMA — chưa kiểm chứng trên tài khoản thật. Mistral OCR trả
`pages[].markdown` là điều chắc chắn; việc nó có kèm bbox + loại block dạng
structured hay không thì TUỲ PHIÊN BẢN, và chưa gọi thử lần nào từ repo này.
Nên adapter dò cả hai dạng:
  - có block structured  -> dùng, ParseResult có boxes, layout_sim dùng được.
  - chỉ có markdown      -> `markdown_to_parse_result`, ParseResult KHÔNG có
    boxes => layout_sim giữa target và Mistral LUÔN = 0 => subtask layout mất
    hẳn model external rẻ, mọi trang tụt xuống Medium/Hard ở riêng subtask đó.
Trường hợp thứ hai không crash nhưng làm hỏng §3.2 ở một subtask, nên
`strict_layout=True` sẽ ném lỗi ngay lần đầu thay vì để nó trôi.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional

from PIL import Image

from ..cmcv import ParseResult
from .base import ClientError, ClientStats, HttpClient
from .cache import DiskCache, image_digest, make_key
from .normalize import (blocks_to_parse_result, is_empty_result,
                        markdown_to_parse_result)
from .openai_compat import encode_image

log = logging.getLogger("ddas.clients.mistral")

MISTRAL_OCR_URL = "https://api.mistral.ai/v1/ocr"
MISTRAL_PROMPT_VERSION = "mistral-ocr-v1"


class MistralOCRRunner:
    def __init__(self, image_fn: Callable[[str], Image.Image], *,
                 model: str = "mistral-ocr-latest",
                 model_name: str = "mistral-ocr-4",
                 url: str = MISTRAL_OCR_URL,
                 api_key_env: str = "MISTRAL_API_KEY",
                 api_key: Optional[str] = None,
                 cache: Optional[DiskCache] = None,
                 rps: float = 4.0,
                 strict_layout: bool = False,
                 allow_empty: bool = False,
                 stats: Optional[ClientStats] = None):
        self.image_fn = image_fn
        self.model = model
        self.model_name = model_name     # phải khớp cmcv.CHEAP_EXTERNAL
        self.url = url
        self.cache = cache
        self.strict_layout = strict_layout
        self.allow_empty = allow_empty
        self.http = HttpClient(model_name, rps=rps, stats=stats)
        key = api_key if api_key is not None else os.environ.get(api_key_env, "")
        if not key:
            log.warning("[%s] thiếu %s — mọi lời gọi sẽ 401", model_name, api_key_env)
        self._headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        self.stats_local = {"pages": 0, "markdown_fallback": 0, "structured": 0, "empty": 0}

    def __call__(self, page_id: str) -> ParseResult:
        img = self.image_fn(page_id)
        ck = make_key(self.model_name, MISTRAL_PROMPT_VERSION,
                      image_digests=[image_digest(img)], params={"model": self.model})
        data = self.cache.get(ck) if self.cache is not None else None
        if data is not None:
            self.http.stats.add(cache_hits=1)
        else:
            payload = {
                "model": self.model,
                "document": {"type": "image_url", "image_url": encode_image(img)},
                "include_image_base64": False,
            }
            t0 = time.monotonic()
            data = self.http.post_json(self.url, payload, self._headers)
            data["_latency_ms"] = (time.monotonic() - t0) * 1000.0
            if self.cache is not None:
                self.cache.put(ck, data)

        pr = self._to_parse_result(page_id, img, data)
        # Cùng lý do với qwen_vl.py: rỗng + rỗng = "đồng thuận" giả.
        if not self.allow_empty and is_empty_result(pr):
            self.stats_local["empty"] += 1
            raise ClientError(
                f"[{self.model_name}] {page_id}: response không có nội dung nào — "
                "cách ly, KHÔNG để thành nhãn Easy rỗng")
        self.stats_local["pages"] += 1
        return pr

    # --------------------------------------------------------------- shape --
    def _to_parse_result(self, page_id: str, img: Image.Image,
                         data: Dict[str, Any]) -> ParseResult:
        pages = data.get("pages") or []
        if not pages:
            raise ClientError(f"[{self.model_name}] {page_id}: response không có 'pages'")
        page = pages[0]
        latency = float(data.get("_latency_ms") or 0.0)
        W, H = img.size

        blocks = _structured_blocks(page)
        if blocks:
            self.stats_local["structured"] += 1
            # Mistral trả toạ độ theo `dimensions` của chính nó, có thể khác
            # kích thước ảnh ta gửi — quy về pixel của ẢNH TA GỬI, vì đó là hệ
            # quy chiếu mà element.crop_box dùng.
            dims = page.get("dimensions") or {}
            sw, sh = int(dims.get("width") or W), int(dims.get("height") or H)
            pr = blocks_to_parse_result(page_id, self.model_name, blocks, sw, sh,
                                        box_space="abs", latency_ms=latency)
            if (sw, sh) != (W, H) and len(pr.boxes):
                pr.boxes[:, [0, 2]] *= W / float(sw)
                pr.boxes[:, [1, 3]] *= H / float(sh)
            return pr

        self.stats_local["markdown_fallback"] += 1
        if self.strict_layout:
            raise ClientError(
                f"[{self.model_name}] {page_id}: response chỉ có markdown, không có bbox. "
                "Với strict_layout=True đây là lỗi: subtask 'layout' của CMCV sẽ mất "
                "external rẻ và mọi trang tụt tier. Kiểm tra lại phiên bản API/tham số, "
                "hoặc tắt strict_layout và chấp nhận layout_sim(target, mistral)=0.")
        if self.stats_local["markdown_fallback"] == 1:
            log.warning("[%s] response không có bbox — ParseResult sẽ không có boxes, "
                        "layout_sim với model này = 0. Xem docstring đầu file.",
                        self.model_name)
        return markdown_to_parse_result(page_id, self.model_name,
                                        page.get("markdown") or "", latency_ms=latency)


def _structured_blocks(page: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Dò block structured ở các tên khoá mà API đã/đang dùng. Rỗng = không có."""
    for key in ("blocks", "elements", "layout", "regions"):
        v = page.get(key)
        if isinstance(v, list) and v and isinstance(v[0], dict):
            if any(k in v[0] for k in ("bbox", "box", "bounding_box")):
                return [_renamed(b) for b in v if isinstance(b, dict)]
    return []


def _renamed(b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(b)
    if "bounding_box" in out and "bbox" not in out:
        bb = out["bounding_box"]
        if isinstance(bb, dict):       # dạng {top_left_x, top_left_y, ...}
            out["bbox"] = [bb.get("top_left_x"), bb.get("top_left_y"),
                           bb.get("bottom_right_x"), bb.get("bottom_right_y")]
        else:
            out["bbox"] = bb
    if "markdown" in out and "content" not in out:
        out["content"] = out["markdown"]
    return out
