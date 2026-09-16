"""Client Gemini 3 Pro — CHỈ dùng cho AI pre-annotation ở §3.3 (paper dòng 64).

Vai trò này được giữ riêng có chủ ý: pool CMCV là Qwen3-VL-8B / Mistral OCR /
PaddleOCR-VL, trọng tài §3.3 là GPT-5, nên Gemini là dòng thứ năm và không đụng
vai nào khác. Đó chính là điều kiện "independence from the CMCV model pool,
thereby avoiding data leakage" mà paper đòi — nếu đem Gemini làm trọng tài luôn
thì bản pre-annotation sẽ kế thừa đúng thiên lệch của bản mà nó vừa chấm.

Giao thức khác OpenAI nên không dùng lại openai_compat.py: Gemini nhận ảnh dưới
dạng inline_data base64 trong `contents[].parts[]`, và system prompt đi ở
`systemInstruction` chứ không phải một message role=system.
"""
from __future__ import annotations

import base64
import logging
import os
import time
from io import BytesIO
from typing import Any, Dict, List, Optional, Sequence

from PIL import Image

from .base import ClientStats, HttpClient
from .cache import DiskCache, image_digest, make_key

log = logging.getLogger("ddas.clients.gemini")

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"


def _inline_image(img: Image.Image, quality: int = 90) -> Dict[str, Any]:
    buf = BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality, optimize=True)
    return {"inline_data": {"mime_type": "image/jpeg",
                            "data": base64.b64encode(buf.getvalue()).decode()}}


class GeminiClient:
    def __init__(self, *, model: str = "gemini-3-pro",
                 base_url: str = GEMINI_BASE,
                 api_key_env: str = "GEMINI_API_KEY",
                 api_key: Optional[str] = None,
                 temperature: float = 0.0,
                 max_tokens: int = 4096,
                 rps: float = 0.0,
                 cache: Optional[DiskCache] = None,
                 stats: Optional[ClientStats] = None):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.cache = cache
        self.http = HttpClient(model, rps=rps, stats=stats)
        self.key = api_key if api_key is not None else os.environ.get(api_key_env, "")
        if not self.key:
            log.warning("[%s] thiếu %s — mọi lời gọi sẽ hỏng", model, api_key_env)

    def call(self, system: str, user_text: str,
             images: Sequence[Image.Image] = (), *,
             prompt_version: str = "", image_digests: Sequence[str] = ()) -> str:
        digests = list(image_digests) or [image_digest(i) for i in images]
        ck = make_key(self.model, prompt_version or "gemini-v1",
                      image_digests=digests, texts=[system, user_text],
                      params={"t": self.temperature, "max_tokens": self.max_tokens})
        if self.cache is not None:
            hit = self.cache.get(ck)
            if hit is not None:
                self.http.stats.add(cache_hits=1)
                return hit.get("text", "")

        parts: List[Dict[str, Any]] = [{"text": user_text}]
        parts.extend(_inline_image(i) for i in images)
        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "systemInstruction": {"parts": [{"text": system}]},
            "generationConfig": {"temperature": self.temperature,
                                 "maxOutputTokens": self.max_tokens},
        }
        url = f"{self.base_url}/models/{self.model}:generateContent"
        t0 = time.monotonic()
        data = self.http.post_json(url, payload, {"x-goog-api-key": self.key,
                                                  "Content-Type": "application/json"})
        text = _extract_text(data)
        usage = data.get("usageMetadata") or {}
        self.http.stats.add(tokens_in=int(usage.get("promptTokenCount") or 0),
                            tokens_out=int(usage.get("candidatesTokenCount") or 0))
        if self.cache is not None:
            self.cache.put(ck, {"text": text, "usage": usage,
                                "latency_ms": (time.monotonic() - t0) * 1000.0})
        return text

    def as_call_model(self, prompt_version: str = "preannot-v1"):
        """Chữ ký `CallModel` mà preannot.make_preannot_fn cần."""
        def call_model(system: str, user_text: str, images: List[Image.Image]) -> str:
            return self.call(system, user_text, images, prompt_version=prompt_version)
        return call_model

    @property
    def stats(self) -> ClientStats:
        return self.http.stats


def _extract_text(data: Dict[str, Any]) -> str:
    for cand in data.get("candidates") or []:
        parts = (cand.get("content") or {}).get("parts") or []
        txt = "".join(p.get("text", "") for p in parts if isinstance(p, dict))
        if txt:
            return txt
    return ""
