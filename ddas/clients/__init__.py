"""Lớp client — nối core DDAS/CMCV/§3.3 vào model thật.

Trước file này, mọi thứ trong repo chạy bằng `ParseResult` giả lập: thuật toán
đúng nhưng chưa từng gọi model nào. Gói `clients/` lấp đúng bốn ranh giới mà
core đã chừa sẵn, không sửa core:

    cmcv.CMCV(runners=...)        <- Dict[str, Callable[[str], ParseResult]]
    judge_refine.make_judge_fn    <- CallModel (system, user, images) -> str
    preannot.make_preannot_fn     <- CallModel
    JudgeRefine(image_fn=...)     <- Callable[[str], Image.Image]

Bản đồ vai -> nhà cung cấp (lý do chọn nằm ở cmcv.py và judge_refine.py):

    TARGET_MODEL       qwen3-vl-8b    tự host (vLLM)      qwen_vl.py
    CHEAP_EXTERNAL     mistral-ocr-4  API, $/trang        mistral_ocr.py
    EXPENSIVE_EXTERNAL paddleocr-vl   tự host             paddle_vl.py
    JUDGE_MODEL §3.3   gpt-5          API, $/token        openai_compat.py
    PREANNOT §3.3      gemini-3-pro   API, $/token        gemini.py

Năm dòng model khác lineage nhau — điều kiện để "đồng thuận" và "trọng tài" có
nghĩa thật, chứ không phải ba biến thể của cùng một họ cùng sai một kiểu.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from PIL import Image

from ..cmcv import (CHEAP_EXTERNAL, EXPENSIVE_EXTERNAL, TARGET_MODEL, CMCVRecord,
                    ParseResult, Tier)
from .base import ClientError, ClientStats, CircuitOpen, RetryPolicy
from .cache import DiskCache
from .normalize import canon_label, to_abs_box
from .pagestore import PageStore

log = logging.getLogger("ddas.clients")

__all__ = ["ClientError", "CircuitOpen", "ClientStats", "RetryPolicy", "DiskCache",
           "PageStore", "canon_label", "to_abs_box", "EngineClients",
           "build_clients", "SafeRunner", "run_cmcv_page", "env_report"]


class SafeRunner:
    """Bọc một runner: lỗi gọi model -> CÁCH LY trang, không trả ParseResult rỗng.

    Lý do tồn tại (xem đầy đủ trong qwen_vl.py): CMCV coi hai ParseResult rỗng
    là đồng thuận hoàn hảo, nên bất kỳ đường nào biến lỗi hạ tầng thành "kết
    quả rỗng" đều sinh ra nhãn Easy rỗng và bơm thẳng vào tập train. Ở quy mô
    triệu trang, một endpoint chập chờn trong 20 phút là đủ để nhiễm bẩn hàng
    chục nghìn mẫu mà không ai nhận ra.

    Ngược lại cũng không được để một trang hỏng làm sập cả job. Nên: nuốt lỗi ở
    mức TRANG, ghi vào `quarantine`, và báo lên trên bằng cách ném tiếp để
    run_cmcv_page() gán tier INVALID.
    """

    def __init__(self, runner: Callable[[str], ParseResult], name: str):
        self.runner = runner
        self.name = name
        self.quarantine: Dict[str, str] = {}      # page_id -> lý do

    def __call__(self, page_id: str) -> ParseResult:
        try:
            return self.runner(page_id)
        except ClientError as e:
            self.quarantine[page_id] = f"{type(e).__name__}: {e}"
            raise
        except (OSError, ValueError) as e:         # ảnh hỏng, không đọc được
            self.quarantine[page_id] = f"{type(e).__name__}: {e}"
            raise ClientError(f"[{self.name}] {page_id}: {e}") from e


def run_cmcv_page(cmcv, page_id: str) -> CMCVRecord:
    """Chạy CMCV một trang, đổi lỗi gọi model thành tier INVALID.

    INVALID là đường đã có sẵn trong core (cmcv.run_page dùng nó cho validity_fn,
    sft.py loại nó khỏi tập train và không đẩy sang §3.3), nên trang hỏng đi
    đúng vào hàng đợi "không dùng được" thay vì tạo nhãn giả.
    """
    try:
        return cmcv.run_page(page_id)
    except ClientError as e:
        log.warning("cách ly %s: %s", page_id, e)
        subtasks = tuple(cmcv.cfg.tau.keys())
        return CMCVRecord(page_id, {t: Tier.INVALID for t in subtasks},
                          {t: {} for t in subtasks}, False, {t: None for t in subtasks})


@dataclass
class EngineClients:
    """Mọi thứ cần để chạy §3.1-§3.3 trên model thật, dựng sẵn và nối đúng vai."""
    page_store: PageStore
    cache: DiskCache
    runners: Dict[str, Callable[[str], ParseResult]] = field(default_factory=dict)
    judge_fn: Optional[Callable] = None
    preannot_fn: Optional[Callable] = None
    _raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def image_fn(self) -> Callable[[str], Image.Image]:
        return self.page_store.as_image_fn()

    def stats(self) -> Dict[str, Any]:
        """Số liệu thật để đối chiếu với dự báo của costmodel.py (mốc M2: ±30%)."""
        out: Dict[str, Any] = {"cache": self.cache.summary(),
                               "page_store": dict(self.page_store.stats)}
        for name, obj in self._raw.items():
            s = getattr(obj, "stats", None)
            if isinstance(s, ClientStats):
                out[name] = s.summary()
            local = getattr(obj, "stats_local", None)
            if isinstance(local, dict):
                out.setdefault(name, {}).update(local)
        for name, r in self.runners.items():
            if isinstance(r, SafeRunner) and r.quarantine:
                out.setdefault(name, {})["quarantined"] = len(r.quarantine)
        return out

    def quarantined(self) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for r in self.runners.values():
            if isinstance(r, SafeRunner):
                out.update(r.quarantine)
        return out


def build_clients(*, pages_root: str,
                  cache_dir: str = ".cache/ddas",
                  qwen_base_url: Optional[str] = None,
                  qwen_model: str = "Qwen/Qwen3-VL-8B-Instruct",
                  paddle_url: Optional[str] = None,
                  mistral_url: Optional[str] = None,
                  judge_model: str = "gpt-5",
                  judge_base_url: Optional[str] = None,
                  preannot_model: str = "gemini-3-pro",
                  preannot_base_url: Optional[str] = None,
                  dpi: int = 200, max_side: int = 1600,
                  mistral_rps: float = 4.0,
                  enable_cache: bool = True,
                  enable: tuple = ("target", "cheap", "expensive", "judge", "preannot"),
                  ) -> EngineClients:
    """Dựng đủ bộ client từ biến môi trường + tham số.

    Vai nào thiếu cấu hình thì BỎ QUA và ghi cảnh báo, không ném lỗi — để chạy
    được từng phần (ví dụ chỉ §3.2 khi chưa có key Gemini). Vai nào thiếu thì
    khâu tương ứng đơn giản là không chạy, chứ không chạy bằng dữ liệu giả.

    MỌI base URL đều ghi đè được (`mistral_url`, `judge_base_url`,
    `preannot_base_url`, hoặc biến môi trường *_BASE_URL tương ứng). Không phải
    tiện nghi: thiếu nó thì test trỏ vào stub localhost vẫn lặng lẽ bắn ra API
    thật của nhà cung cấp — đúng thứ đã xảy ra một lần khi dựng preflight.
    """
    store = PageStore(pages_root, dpi=dpi, max_side=max_side)
    cache = DiskCache(cache_dir, enabled=enable_cache)
    image_fn = store.as_image_fn()
    runners: Dict[str, Callable[[str], ParseResult]] = {}
    raw: Dict[str, Any] = {}

    if "target" in enable:
        url = qwen_base_url or os.environ.get("QWEN_BASE_URL", "")
        if url:
            from .qwen_vl import build_qwen_runner
            r = build_qwen_runner(url, qwen_model, image_fn,
                                  model_name=TARGET_MODEL, cache=cache)
            runners[TARGET_MODEL] = SafeRunner(r, TARGET_MODEL)
            raw[TARGET_MODEL] = r.client
        else:
            log.warning("thiếu QWEN_BASE_URL -> không có TARGET_MODEL, CMCV không chạy được")

    if "cheap" in enable:
        if os.environ.get("MISTRAL_API_KEY"):
            from .mistral_ocr import MISTRAL_OCR_URL, MistralOCRRunner
            r = MistralOCRRunner(image_fn, model_name=CHEAP_EXTERNAL,
                                 url=(mistral_url or os.environ.get("MISTRAL_OCR_URL")
                                      or MISTRAL_OCR_URL),
                                 cache=cache, rps=mistral_rps)
            runners[CHEAP_EXTERNAL] = SafeRunner(r, CHEAP_EXTERNAL)
            raw[CHEAP_EXTERNAL] = r
        else:
            log.warning("thiếu MISTRAL_API_KEY -> không có CHEAP_EXTERNAL")

    if "expensive" in enable:
        url = paddle_url or os.environ.get("PADDLE_VL_URL", "")
        if url:
            from .paddle_vl import PaddleVLRunner
            r = PaddleVLRunner(url, image_fn, model_name=EXPENSIVE_EXTERNAL, cache=cache)
            runners[EXPENSIVE_EXTERNAL] = SafeRunner(r, EXPENSIVE_EXTERNAL)
            raw[EXPENSIVE_EXTERNAL] = r
        else:
            log.warning("thiếu PADDLE_VL_URL -> không có EXPENSIVE_EXTERNAL; "
                        "cascade sẽ không có trọng tài, mọi bất đồng target/cheap "
                        "rơi thẳng xuống Hard")

    judge_fn = preannot_fn = None
    if "judge" in enable:
        if os.environ.get("OPENAI_API_KEY"):
            from ..judge_refine import make_judge_fn
            from .openai_compat import OpenAICompatClient, OpenAICompatConfig
            c = OpenAICompatClient(
                OpenAICompatConfig(model=judge_model,
                                   base_url=(judge_base_url
                                             or os.environ.get("OPENAI_BASE_URL")
                                             or "https://api.openai.com/v1"),
                                   max_tokens=8192), cache)
            judge_fn = make_judge_fn(c.as_call_model("judge-v1"))
            raw[judge_model] = c
        else:
            log.warning("thiếu OPENAI_API_KEY -> §3.3 không có trọng tài")

    if "preannot" in enable:
        if os.environ.get("GEMINI_API_KEY"):
            from ..preannot import make_preannot_fn
            from .gemini import GEMINI_BASE, GeminiClient
            g = GeminiClient(model=preannot_model, cache=cache,
                             base_url=(preannot_base_url
                                       or os.environ.get("GEMINI_BASE_URL") or GEMINI_BASE))
            preannot_fn = make_preannot_fn(g.as_call_model("preannot-v1"))
            raw[preannot_model] = g
        else:
            log.warning("thiếu GEMINI_API_KEY -> §3.3 không có AI pre-annotation")

    return EngineClients(store, cache, runners, judge_fn, preannot_fn, raw)


def env_report() -> Dict[str, bool]:
    """Kiểm nhanh cấu hình môi trường trước khi chạy đợt tốn tiền."""
    return {
        "QWEN_BASE_URL (target)": bool(os.environ.get("QWEN_BASE_URL")),
        "MISTRAL_API_KEY (cheap external)": bool(os.environ.get("MISTRAL_API_KEY")),
        "PADDLE_VL_URL (expensive external)": bool(os.environ.get("PADDLE_VL_URL")),
        "OPENAI_API_KEY (judge §3.3)": bool(os.environ.get("OPENAI_API_KEY")),
        "GEMINI_API_KEY (pre-annotation §3.3)": bool(os.environ.get("GEMINI_API_KEY")),
    }
