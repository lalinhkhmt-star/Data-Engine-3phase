"""Client cho mọi endpoint nói giao thức OpenAI chat-completions.

Một lớp dùng cho hai vai KHÁC HẲN nhau về tính chất, cố ý gộp vì giao thức
giống nhau — nhưng đừng nhầm hai vai đó:

  - Qwen3-VL-8B TỰ HOST qua vLLM (`--api-key` tuỳ chọn, base_url nội bộ). Đây
    là TARGET_MODEL của §3.2: model đang được cải thiện, output của nó trên
    trang Easy trở thành nhãn SFT. Chi phí tính bằng GPU-giờ, không phải USD.
  - GPT-5 qua API OpenAI. Đây là JUDGE_MODEL của §3.3, tính tiền theo token và
    là dòng model thứ tư (khác cả ba model trong pool CMCV — xem judge_refine.py).

Hàm `call_model` trả về đúng chữ ký `CallModel` mà judge_refine.py/preannot.py
cần: (system, user_text, images) -> text thô. Ranh giới nhà cung cấp nằm gọn ở
đây, prompt và vòng lặp §3.3 không biết gì về HTTP.
"""
from __future__ import annotations

import base64
import logging
import os
import time
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, List, Optional, Sequence

from PIL import Image

from .base import ClientStats, HttpClient
from .cache import DiskCache, image_digest, make_key

log = logging.getLogger("ddas.clients.openai")


def encode_image(img: Image.Image, fmt: str = "JPEG", quality: int = 90) -> str:
    """PIL -> data URL. JPEG q90 cho ảnh scan: nhỏ hơn PNG ~5-10 lần khi gửi
    qua mạng, và ở 90 thì nhiễu nén không ảnh hưởng tới OCR. Ảnh đã được
    PageStore hạ về max_side trước đó (xem pagestore.py)."""
    buf = BytesIO()
    if fmt.upper() == "JPEG":
        img.convert("RGB").save(buf, format="JPEG", quality=quality, optimize=True)
        mime = "image/jpeg"
    else:
        img.save(buf, format="PNG")
        mime = "image/png"
    return f"data:{mime};base64," + base64.b64encode(buf.getvalue()).decode()


@dataclass
class OpenAICompatConfig:
    model: str
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    api_key: Optional[str] = None       # ưu tiên hơn api_key_env; vLLM có thể bỏ trống
    temperature: float = 0.0            # bóc tách/trọng tài là tác vụ tất định
    max_tokens: int = 4096
    timeout: float = 240.0
    rps: float = 0.0                    # 0 = không tự phanh (endpoint tự host)
    image_format: str = "JPEG"
    json_mode: bool = True              # ép response_format=json_object khi endpoint hỗ trợ
    extra_body: Dict[str, Any] = None   # tham số riêng của backend (vLLM: top_k, ...)


class OpenAICompatClient:
    """Gọi /chat/completions có ảnh, qua HttpClient (retry/rate-limit/breaker) + cache."""

    def __init__(self, cfg: OpenAICompatConfig, cache: Optional[DiskCache] = None,
                 stats: Optional[ClientStats] = None):
        self.cfg = cfg
        self.cache = cache
        self.http = HttpClient(cfg.model, timeout=cfg.timeout, rps=cfg.rps, stats=stats)
        key = cfg.api_key if cfg.api_key is not None else os.environ.get(cfg.api_key_env, "")
        self._auth = {"Authorization": f"Bearer {key}"} if key else {}
        if not key and "api.openai.com" in cfg.base_url:
            log.warning("[%s] thiếu %s — gọi tới OpenAI sẽ bị 401",
                        cfg.model, cfg.api_key_env)

    # ------------------------------------------------------------- public --
    def call(self, system: str, user_text: str,
             images: Sequence[Image.Image] = (), *,
             prompt_version: str = "", image_digests: Sequence[str] = ()) -> str:
        digests = list(image_digests) or [image_digest(i) for i in images]
        ck = make_key(self.cfg.model, prompt_version or _hash_prompt(system, user_text),
                      image_digests=digests, texts=[system, user_text],
                      params={"t": self.cfg.temperature, "max_tokens": self.cfg.max_tokens})
        if self.cache is not None:
            hit = self.cache.get(ck)
            if hit is not None:
                self.http.stats.add(cache_hits=1)
                return hit.get("text", "")

        content: List[Dict[str, Any]] = [{"type": "text", "text": user_text}]
        for img in images:
            content.append({"type": "image_url",
                            "image_url": {"url": encode_image(img, self.cfg.image_format)}})
        payload: Dict[str, Any] = {
            "model": self.cfg.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": content}],
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
        }
        if self.cfg.json_mode:
            payload["response_format"] = {"type": "json_object"}
        if self.cfg.extra_body:
            payload.update(self.cfg.extra_body)

        t0 = time.monotonic()
        data = self.http.post_json(f"{self.cfg.base_url.rstrip('/')}/chat/completions",
                                   payload, self._auth)
        latency_ms = (time.monotonic() - t0) * 1000.0
        text = _extract_text(data)
        usage = data.get("usage") or {}
        self.http.stats.add(tokens_in=int(usage.get("prompt_tokens") or 0),
                            tokens_out=int(usage.get("completion_tokens") or 0))
        if self.cache is not None:
            self.cache.put(ck, {"text": text, "latency_ms": latency_ms,
                                "model": self.cfg.model, "usage": usage})
        return text

    def as_call_model(self, prompt_version: str = ""):
        """Trả về `CallModel` đúng chữ ký judge_refine.py/preannot.py cần."""
        def call_model(system: str, user_text: str, images: List[Image.Image]) -> str:
            return self.call(system, user_text, images, prompt_version=prompt_version)
        return call_model

    @property
    def stats(self) -> ClientStats:
        return self.http.stats


def _extract_text(data: Dict[str, Any]) -> str:
    choices = data.get("choices") or []
    if not choices:
        return ""
    msg = choices[0].get("message") or {}
    c = msg.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):        # một số backend trả content dạng mảng part
        return "".join(p.get("text", "") for p in c if isinstance(p, dict))
    return ""


def _hash_prompt(system: str, user_text: str) -> str:
    import hashlib
    return hashlib.sha256((system + "\x00" + user_text).encode()).hexdigest()[:12]
