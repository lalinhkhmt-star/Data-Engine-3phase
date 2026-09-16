"""Runner CMCV cho VLM tổng quát — mặc định Qwen3-VL-8B (TARGET_MODEL, tự host).

Trả về đúng `Callable[[str], ParseResult]` mà cmcv.CMCV cần.

Điểm phải cẩn thận nhất ở đây KHÔNG phải là gọi được API, mà là xử lý output
hỏng. Chữ ký runner của CMCV không có kênh báo lỗi (`page_id -> ParseResult`),
nên cám dỗ tự nhiên là trả về ParseResult rỗng khi model lỗi. Làm vậy là hỏng
âm thầm ở mức tệ nhất có thể:

    ParseResult rỗng  ~  ParseResult rỗng   =>  mọi sim = 1.0  =>  tier EASY
    => trang đó vào tập train với nhãn RỖNG, được đánh dấu là "model đồng thuận"

Tức là lỗi hạ tầng bị biến thành dữ liệu huấn luyện sai, và không có chỗ nào
trong pipeline phát hiện ra. Nên ở đây output không parse được thì NÉM LỖI
(`ParseFailed`), để khâu điều phối cách ly trang đó (xem clients/__init__.py::
SafeRunner) thay vì lùa nó vào tier Easy.
"""
from __future__ import annotations

import logging
import time
from typing import Callable, Optional, Sequence

from PIL import Image

from ..cmcv import ParseResult
from ..prompts import (PARSE_BOX_SPACE, PARSE_PROMPT_VERSION, PARSE_SYSTEM,
                       parse_blocks, parse_user_prompt)
from .base import ClientError
from .cache import DiskCache, image_digest
from .normalize import blocks_to_parse_result, is_empty_result
from .openai_compat import OpenAICompatClient

log = logging.getLogger("ddas.clients.qwen")


class ParseFailed(ClientError):
    """Model trả về thứ không bóc thành block được — KHÔNG phải trang trắng."""


class VLMPageRunner:
    """Bóc tách trang bằng VLM tổng quát qua giao thức OpenAI chat-completions."""

    def __init__(self, client: OpenAICompatClient, model_name: str,
                 image_fn: Callable[[str], Image.Image],
                 *, box_space: str = PARSE_BOX_SPACE, retry_unparseable: int = 1,
                 allow_empty: bool = False):
        self.client = client
        self.model_name = model_name       # tên ghi vào ParseResult.model (cmcv.TARGET_MODEL)
        self.image_fn = image_fn
        self.box_space = box_space
        # Output không đúng JSON là chuyện thỉnh thoảng xảy ra với VLM kể cả khi
        # bật json_mode. Thử lại 1 lần với lời nhắc cứng hơn rẻ hơn nhiều so với
        # đẩy trang sang hàng đợi cách ly.
        self.retry_unparseable = retry_unparseable
        # Xem chặn trang rỗng ở cuối __call__ — mặc định KHÔNG cho qua.
        self.allow_empty = allow_empty
        self.stats = {"pages": 0, "unparseable": 0, "reparse_ok": 0, "empty": 0}

    def __call__(self, page_id: str) -> ParseResult:
        img = self.image_fn(page_id)
        digests = [image_digest(img)]
        user = parse_user_prompt()
        t0 = time.monotonic()

        blocks = None
        for attempt in range(self.retry_unparseable + 1):
            if attempt:
                # Đổi prompt => đổi khoá cache, nên lần thử lại KHÔNG trúng ô
                # cache của lần hỏng trước (xem cache.py).
                user = (parse_user_prompt() +
                        "\n\nLƯU Ý: lần trước bạn trả về sai định dạng. "
                        "Chỉ được trả về DUY NHẤT một object JSON có khoá "
                        '"blocks", không kèm bất cứ chữ nào khác.')
            raw = self.client.call(PARSE_SYSTEM, user, [img],
                                   prompt_version=f"{PARSE_PROMPT_VERSION}+{attempt}",
                                   image_digests=digests)
            blocks = parse_blocks(raw)
            if blocks is not None:
                if attempt:
                    self.stats["reparse_ok"] += 1
                break
            self.stats["unparseable"] += 1
            log.warning("[%s] %s: output không parse được (lần %d): %.200s",
                        self.model_name, page_id, attempt + 1, raw)

        if blocks is None:
            raise ParseFailed(
                f"[{self.model_name}] {page_id}: không bóc được block sau "
                f"{self.retry_unparseable + 1} lần — cách ly trang, KHÔNG coi là trang trắng")

        W, H = img.size
        pr = blocks_to_parse_result(page_id, self.model_name, blocks, W, H,
                                    box_space=self.box_space,
                                    latency_ms=(time.monotonic() - t0) * 1000.0)

        # Trang không có một mẩu nội dung nào: có thể là trang trắng THẬT, có thể
        # là model bó tay. Không phân biệt được từ đây, và cũng KHÔNG cần: trang
        # trắng không mang tín hiệu huấn luyện cho bất kỳ subtask nào trong bốn
        # subtask, nên loại bỏ nó không mất gì. Ngược lại, cho nó đi tiếp thì ba
        # kết quả rỗng "đồng thuận" với nhau và sinh nhãn EASY rỗng (xem
        # normalize.is_empty_result). Cổng đúng cho trang trắng là scanqa.py,
        # chạy TRƯỚC mọi lời gọi model; đây chỉ là lớp chặn thứ hai.
        if not self.allow_empty and is_empty_result(pr):
            self.stats["empty"] += 1
            raise ParseFailed(
                f"[{self.model_name}] {page_id}: không có nội dung nào (trang trắng "
                "hoặc model bó tay) — cách ly, KHÔNG để thành nhãn Easy rỗng")

        self.stats["pages"] += 1
        return pr


def build_qwen_runner(base_url: str, model: str, image_fn: Callable[[str], Image.Image],
                      *, model_name: str = "qwen3-vl-8b",
                      api_key: Optional[str] = None,
                      cache: Optional[DiskCache] = None,
                      rps: float = 0.0, max_tokens: int = 6144) -> VLMPageRunner:
    """Dựng runner trỏ vào endpoint vLLM tự host.

    `model` là tên model phía vLLM (ví dụ "Qwen/Qwen3-VL-8B-Instruct"), còn
    `model_name` là tên trong hệ thống này (phải khớp cmcv.TARGET_MODEL, vì
    sft.py tra nguồn nhãn theo đúng chuỗi đó).
    """
    from .openai_compat import OpenAICompatConfig

    cfg = OpenAICompatConfig(model=model, base_url=base_url, api_key=api_key or "",
                             max_tokens=max_tokens, rps=rps,
                             # vLLM hỗ trợ json_object không đồng đều theo phiên
                             # bản; parse_blocks đã chịu được output có rác bọc
                             # ngoài nên không phụ thuộc vào nó.
                             json_mode=False)
    return VLMPageRunner(OpenAICompatClient(cfg, cache), model_name, image_fn)
