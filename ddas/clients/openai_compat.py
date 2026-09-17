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
    # bóc tách/trọng tài là tác vụ tất định nên mặc định 0.0 — nhưng model
    # reasoning đời mới của OpenAI (GPT-5, o1, o3, ...) TỪ CHỐI mọi giá trị
    # khác mặc định (400 "Unsupported value... Only the default (1) value is
    # supported"), đo thật lúc preflight. None = KHÔNG gửi tham số này, để
    # API tự dùng mặc định của nó — không hardcode 1.0 vì mặc định có thể đổi
    # theo model/phiên bản, "không gửi" luôn an toàn hơn "đoán đúng số".
    temperature: Optional[float] = 0.0
    max_tokens: int = 4096
    # Tên tham số giới hạn token ra — KHÁC NHAU giữa nhà cung cấp, không phải
    # tuỳ chọn thẩm mỹ: vLLM (Qwen tự host) nhận "max_tokens"; OpenAI với các
    # model reasoning đời mới (GPT-5, o1, o3, ...) BẮT BUỘC "max_completion_tokens"
    # và trả lỗi 400 nếu gửi "max_tokens" — đã đo thật lúc preflight trên GPT-5.
    # Không tự đoán theo tên model (danh sách model reasoning còn thay đổi) —
    # người gọi khai rõ theo từng provider, xem clients/__init__.py.
    max_tokens_param: str = "max_tokens"
    timeout: float = 240.0
    rps: float = 0.0                    # 0 = không tự phanh (endpoint tự host)
    image_format: str = "JPEG"
    json_mode: bool = True              # ép response_format=json_object khi endpoint hỗ trợ
    # Schema BẮT BUỘC cho output, dạng {"name": ..., "schema": {...}}. Khác
    # json_mode ở chỗ: json_mode chỉ ép "trả JSON hợp lệ", KHÔNG ép đúng schema —
    # ĐO THẬT trên gpt-5 và gpt-5-mini: cả hai đều bỏ trường `confidence` dù
    # schema khai nó là required, và parse_verdict() âm thầm biến thành 0.0,
    # làm chết tiêu chí ưu tiên #1 của §3.3 (prioritize() cần confidence >=
    # min_confidence). Đặt json_schema thì OpenAI ép đúng schema (strict).
    json_schema: Optional[Dict[str, Any]] = None
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
                      # json_schema PHẢI nằm trong khoá: bật/tắt nó đổi hẳn output
                      # (ép đúng schema hay không), quên thì lần bật đầu tiên sẽ
                      # trúng cache cũ và tưởng là fix không có tác dụng.
                      params={"t": self.cfg.temperature,
                              self.cfg.max_tokens_param: self.cfg.max_tokens,
                              "schema": (self.cfg.json_schema or {}).get("name")})
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
            self.cfg.max_tokens_param: self.cfg.max_tokens,
        }
        if self.cfg.temperature is not None:
            payload["temperature"] = self.cfg.temperature
        if self.cfg.json_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"strict": True, **self.cfg.json_schema},
            }
        elif self.cfg.json_mode:
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
