"""Hạ tầng gọi model — retry, rate-limit, circuit-breaker, đếm chi phí.

Mọi client trong gói này đi qua đúng một lớp HTTP để bốn thứ dưới đây được áp
dụng đồng nhất, không phải nhớ lặp lại ở từng nhà cung cấp:

  1. RETRY có phân biệt lỗi. 429/5xx/timeout là lỗi TẠM THỜI -> thử lại với
     backoff luỹ thừa + jitter. 400/401/404 là lỗi CỦA TA (prompt sai, key sai,
     endpoint sai) -> thử lại chỉ tốn tiền và thời gian, nên fail ngay.
  2. RATE-LIMIT chủ động (token bucket). costmodel.py cố ý KHÔNG mô hình hoá
     rate-limit nhà cung cấp; ở quy mô hàng triệu trang thì đây là thứ quyết
     định wall-clock thật, và đụng trần phía nhà cung cấp sẽ nhận 429 hàng loạt.
     Tự phanh ở phía mình rẻ hơn là để họ phanh hộ.
  3. CIRCUIT-BREAKER. Endpoint chết mà vẫn bắn đủ số retry cho từng trang trong
     hàng đợi triệu mẫu thì mất hàng giờ chỉ để tích lỗi. N lần hỏng liên tiếp
     -> mở mạch, fail nhanh trong `open_seconds`, rồi thử lại một lời gọi thăm dò.
  4. ĐẾM CHI PHÍ THẬT. `ClientStats` đếm lời gọi/token/lỗi/cache-hit theo model,
     để đối chiếu với dự báo trong costmodel.py — mốc M2 cần khớp ±30%.

Luồng chạy dài ngày nên mọi lỗi đều ghi lại được, không nuốt im lặng.
"""
from __future__ import annotations

import logging
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import requests

log = logging.getLogger("ddas.clients")

# Mã lỗi coi là TẠM THỜI — đáng thử lại.
RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})


class ClientError(RuntimeError):
    """Lỗi gọi model đã hết đường cứu (hết retry, hoặc lỗi không đáng retry)."""

    def __init__(self, msg: str, status: Optional[int] = None, retryable: bool = False):
        super().__init__(msg)
        self.status = status
        self.retryable = retryable


class CircuitOpen(ClientError):
    """Mạch đang mở — fail nhanh, không gọi mạng."""


@dataclass
class RetryPolicy:
    max_attempts: int = 5
    base_delay: float = 1.0         # giây; delay = base * 2^(lần thử - 1)
    max_delay: float = 60.0
    jitter: float = 0.3             # +-30% để tránh thundering herd khi chạy song song

    def delay_for(self, attempt: int) -> float:
        d = min(self.base_delay * (2 ** max(0, attempt - 1)), self.max_delay)
        return d * (1.0 + random.uniform(-self.jitter, self.jitter))


class RateLimiter:
    """Token bucket thread-safe. `rps<=0` => không giới hạn."""

    def __init__(self, rps: float, burst: Optional[float] = None):
        self.rps = float(rps)
        self.capacity = float(burst if burst is not None else max(1.0, rps))
        self._tokens = self.capacity
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, n: float = 1.0) -> float:
        """Chặn tới khi có đủ token. Trả về số giây đã phải chờ."""
        if self.rps <= 0:
            return 0.0
        waited = 0.0
        while True:
            with self._lock:
                now = time.monotonic()
                self._tokens = min(self.capacity, self._tokens + (now - self._last) * self.rps)
                self._last = now
                if self._tokens >= n:
                    self._tokens -= n
                    return waited
                need = (n - self._tokens) / self.rps
            time.sleep(need)
            waited += need


class CircuitBreaker:
    """Đóng/mở theo số lần hỏng LIÊN TIẾP. Một lần thành công là reset."""

    def __init__(self, threshold: int = 8, open_seconds: float = 60.0):
        self.threshold = threshold
        self.open_seconds = open_seconds
        self._fails = 0
        self._opened_at = 0.0
        self._lock = threading.Lock()

    def check(self, name: str) -> None:
        with self._lock:
            if self._fails < self.threshold:
                return
            if time.monotonic() - self._opened_at < self.open_seconds:
                raise CircuitOpen(
                    f"[{name}] mạch mở sau {self._fails} lỗi liên tiếp; "
                    f"thử lại sau {self.open_seconds:.0f}s", retryable=True)
            # Hết thời gian mở -> cho MỘT lời gọi thăm dò đi qua.
            self._fails = self.threshold - 1

    def record(self, ok: bool) -> None:
        with self._lock:
            if ok:
                self._fails = 0
            else:
                self._fails += 1
                if self._fails == self.threshold:
                    self._opened_at = time.monotonic()


@dataclass
class ClientStats:
    """Đếm theo model — đối chiếu với dự báo của costmodel.py."""
    calls: int = 0
    cache_hits: int = 0
    errors: int = 0
    retries: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    seconds: float = 0.0
    rate_limit_wait: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, **kw: float) -> None:
        with self._lock:
            for k, v in kw.items():
                setattr(self, k, getattr(self, k) + v)

    @property
    def cache_hit_rate(self) -> float:
        total = self.calls + self.cache_hits
        return self.cache_hits / total if total else 0.0

    def summary(self) -> Dict[str, float]:
        return {"calls": self.calls, "cache_hits": self.cache_hits,
                "cache_hit_rate": round(self.cache_hit_rate, 4),
                "errors": self.errors, "retries": self.retries,
                "tokens_in": self.tokens_in, "tokens_out": self.tokens_out,
                "avg_latency_s": round(self.seconds / self.calls, 3) if self.calls else 0.0,
                "rate_limit_wait_s": round(self.rate_limit_wait, 1)}


class HttpClient:
    """POST JSON có retry/rate-limit/circuit-breaker. Dùng chung cho mọi provider."""

    def __init__(self, name: str, *, timeout: float = 180.0,
                 rps: float = 0.0, burst: Optional[float] = None,
                 retry: Optional[RetryPolicy] = None,
                 breaker: Optional[CircuitBreaker] = None,
                 stats: Optional[ClientStats] = None):
        self.name = name
        self.timeout = timeout
        self.limiter = RateLimiter(rps, burst)
        self.retry = retry or RetryPolicy()
        self.breaker = breaker or CircuitBreaker()
        self.stats = stats or ClientStats()
        self.session = requests.Session()

    def post_json(self, url: str, payload: Dict[str, Any],
                  headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        last: Optional[Exception] = None
        for attempt in range(1, self.retry.max_attempts + 1):
            self.breaker.check(self.name)
            self.stats.add(rate_limit_wait=self.limiter.acquire())
            t0 = time.monotonic()
            try:
                r = self.session.post(url, json=payload, headers=headers or {},
                                      timeout=self.timeout)
                dt = time.monotonic() - t0
                if r.status_code >= 400:
                    retryable = r.status_code in RETRYABLE_STATUS
                    raise ClientError(f"[{self.name}] HTTP {r.status_code}: {r.text[:300]}",
                                      r.status_code, retryable)
                self.breaker.record(True)
                self.stats.add(calls=1, seconds=dt)
                return r.json()
            except (requests.Timeout, requests.ConnectionError) as e:
                last = ClientError(f"[{self.name}] {type(e).__name__}: {e}", None, True)
            except ClientError as e:
                last = e
                if not e.retryable:          # lỗi của ta -> retry chỉ tốn tiền
                    self.breaker.record(True)
                    self.stats.add(errors=1)
                    raise
            except ValueError as e:          # body không phải JSON hợp lệ
                last = ClientError(f"[{self.name}] body không phải JSON: {e}", None, True)

            self.breaker.record(False)
            if attempt < self.retry.max_attempts:
                d = self.retry.delay_for(attempt)
                self.stats.add(retries=1)
                log.warning("[%s] lần %d/%d hỏng (%s) — chờ %.1fs",
                            self.name, attempt, self.retry.max_attempts, last, d)
                time.sleep(d)

        self.stats.add(errors=1)
        raise ClientError(f"[{self.name}] hỏng sau {self.retry.max_attempts} lần: {last}")
