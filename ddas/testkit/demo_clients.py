"""Kiểm chứng lớp client end-to-end qua HTTP thật trên localhost (không tốn tiền).

    python3 -m ddas.testkit.demo_clients

Bảy bài, mỗi bài kiểm một thứ có thể hỏng ÂM THẦM ở production:

  1. Đường sạch     — 3 runner trả ParseResult hợp lệ, bbox về đúng pixel ảnh.
  2. CMCV + cascade — 3 model khớp nhau => EASY, và model thứ 3 KHÔNG bị gọi.
  3. Tier MEDIUM    — target lệch, 2 external khớp => MEDIUM, nhãn lấy từ cheap.
  4. Cache          — chạy lần 2 phải 0 lời gọi mạng mới.
  5. Retry          — 503 hai lần đầu vẫn ra kết quả đúng.
  6. Output hỏng    — KHÔNG được thành ParseResult rỗng; phải rơi vào INVALID.
  7. Judge §3.3     — make_judge_fn qua HTTP thật, parse verdict đúng.

Bài 6 là bài quan trọng nhất: nó chặn đúng con đường biến lỗi hạ tầng thành
nhãn huấn luyện sai (xem qwen_vl.py).
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image, ImageDraw

from ..cmcv import CHEAP_EXTERNAL, CMCV, EXPENSIVE_EXTERNAL, TARGET_MODEL, Tier
from ..clients import DiskCache, PageStore, SafeRunner, run_cmcv_page
from ..clients.mistral_ocr import MistralOCRRunner
from ..clients.paddle_vl import PaddleVLRunner
from ..clients.qwen_vl import build_qwen_runner
from ..config import CMCVConfig
from .stub_server import DEFAULT_BLOCKS, ServerScript, StubServer

PASS, FAIL = "  ✓", "  ✗"
_results: List[bool] = []


def block_external_network() -> None:
    """Chặn cứng mọi kết nối ra ngoài localhost trong suốt bài test.

    Không phải phòng xa: khi dựng preflight, một lần cấu hình thiếu base URL đã
    khiến test trỏ vào stub localhost vẫn lặng lẽ bắn ảnh ra api.mistral.ai,
    api.openai.com và googleapis.com với key giả. Chúng chỉ trả 401 nên không
    có gì đổ vỡ và KHÔNG có dấu hiệu nào trong output — đúng kiểu lỗi chỉ lộ ra
    khi có người ngồi đọc log. Chốt này biến nó thành lỗi ồn ào ngay lập tức.
    """
    import socket
    real_connect = socket.socket.connect

    def guarded(self, address):
        host = address[0] if isinstance(address, tuple) else None
        if host not in ("127.0.0.1", "::1", "localhost", None):
            raise AssertionError(
                f"test offline nhưng đang kết nối ra ngoài: {address}. "
                "Kiểm tra base_url của client có trỏ về stub không.")
        return real_connect(self, address)

    socket.socket.connect = guarded


def check(cond: bool, msg: str) -> bool:
    _results.append(bool(cond))
    print(f"{PASS if cond else FAIL} {msg}")
    return bool(cond)


def make_page(path: Path, seed: int = 0) -> None:
    """Trang giả 1000x1400 — chỉ cần bytes ảnh ổn định để test, không cần đẹp."""
    img = Image.new("RGB", (1000, 1400), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([80, 50, 920, 140], outline="black", width=3)
    d.text((100, 80), f"BAO CAO TAI CHINH QUY III - trang {seed}", fill="black")
    for i in range(12):
        d.line([100, 200 + i * 20, 900, 200 + i * 20], fill="gray")
    img.save(path)


def build_runners(url: str, image_fn, cache: DiskCache) -> Dict[str, SafeRunner]:
    q = build_qwen_runner(f"{url}/v1", "qwen-stub", image_fn,
                          model_name=TARGET_MODEL, api_key="stub", cache=cache)
    m = MistralOCRRunner(image_fn, model_name=CHEAP_EXTERNAL, url=f"{url}/v1/ocr",
                         api_key="stub", cache=cache, rps=0.0)
    p = PaddleVLRunner(f"{url}/layout-parsing", image_fn,
                       model_name=EXPENSIVE_EXTERNAL, cache=cache)
    return {TARGET_MODEL: SafeRunner(q, TARGET_MODEL),
            CHEAP_EXTERNAL: SafeRunner(m, CHEAP_EXTERNAL),
            EXPENSIVE_EXTERNAL: SafeRunner(p, EXPENSIVE_EXTERNAL)}


def main() -> int:
    block_external_network()
    root = Path(tempfile.mkdtemp(prefix="ddas_clients_"))
    try:
        pages = root / "pages"
        pages.mkdir()
        for i in range(3):
            make_page(pages / f"p{i}.png", i)
        store = PageStore(pages, max_side=1000)
        image_fn = store.as_image_fn()

        # ------------------------------------------------ 1. đường sạch ----
        print("\n[1] Đường sạch — 3 runner trả ParseResult hợp lệ")
        script = ServerScript()
        with StubServer(script) as url:
            cache = DiskCache(root / "c1")
            runners = build_runners(url, image_fn, cache)
            prs = {name: r("p0.png") for name, r in runners.items()}
            for name, pr in prs.items():
                ok = (pr.model == name and len(pr.boxes) == 4
                      and len(pr.tables) == 1 and len(pr.formulas) == 1)
                check(ok, f"{name}: {len(pr.boxes)} box, {len(pr.tables)} bảng, "
                          f"{len(pr.formulas)} công thức")
            W, H = image_fn("p0.png").size
            b = prs[TARGET_MODEL].boxes
            check(bool(b[:, 2].max() <= W and b[:, 3].max() <= H),
                  f"bbox nằm trong ảnh {W}x{H} (max x={b[:,2].max():.0f}, y={b[:,3].max():.0f})")
            check(prs[TARGET_MODEL].labels[:2] == ["title", "text"],
                  f"nhãn quy về taxonomy chuẩn: {prs[TARGET_MODEL].labels}")

        # -------------------------------------------- 2. CMCV + cascade ----
        print("\n[2] CMCV — 3 model khớp => EASY, cascade cắt model thứ 3")
        script = ServerScript()
        with StubServer(script) as url:
            cache = DiskCache(root / "c2")
            runners = build_runners(url, image_fn, cache)
            cmcv = CMCV(runners, CMCVConfig())
            rec = run_cmcv_page(cmcv, "p0.png")
            check(all(t == Tier.EASY for t in rec.tier.values()),
                  f"mọi subtask EASY: {dict((k, v.value) for k, v in rec.tier.items())}")
            check(script.calls.get("paddle", 0) == 0,
                  f"cascade cắt PaddleOCR-VL: {script.calls.get('paddle', 0)} lời gọi")
            check(rec.pseudo_label_from["text"] == TARGET_MODEL,
                  f"nhãn Easy lấy từ target: {rec.pseudo_label_from['text']}")

        # ------------------------------------------------ 3. tier MEDIUM ---
        print("\n[3] Target lệch, 2 external khớp => MEDIUM")
        diverged = [dict(b) for b in DEFAULT_BLOCKS]
        diverged[1] = dict(diverged[1],
                           content="Doanh thu thuan tang truong so voi cung ky nam truoc")
        script = ServerScript(blocks={"qwen-stub": diverged})
        with StubServer(script) as url:
            cache = DiskCache(root / "c3")
            runners = build_runners(url, image_fn, cache)
            rec = run_cmcv_page(CMCV(runners, CMCVConfig()), "p1.png")
            check(rec.tier["text"] == Tier.MEDIUM,
                  f"text = {rec.tier['text'].value} (target mất dấu, 2 external khớp)")
            check(rec.pseudo_label_from["text"] == CHEAP_EXTERNAL,
                  f"nhãn Medium lấy từ external: {rec.pseudo_label_from['text']}")
            check(script.calls.get("paddle", 0) == 1,
                  "cascade GỌI model thứ 3 khi có bất đồng")

        # ----------------------------------------------------- 4. cache ----
        print("\n[4] Cache — chạy lại không gọi mạng")
        script = ServerScript()
        with StubServer(script) as url:
            cache = DiskCache(root / "c4")
            runners = build_runners(url, image_fn, cache)
            run_cmcv_page(CMCV(runners, CMCVConfig()), "p2.png")
            n1 = sum(script.calls.values())
            runners2 = build_runners(url, image_fn, cache)   # client MỚI, cache cũ
            rec2 = run_cmcv_page(CMCV(runners2, CMCVConfig()), "p2.png")
            n2 = sum(script.calls.values())
            check(n2 == n1, f"lần 2 thêm {n2 - n1} lời gọi mạng (phải = 0)")
            check(all(t == Tier.EASY for t in rec2.tier.values()),
                  "kết quả từ cache giống hệt lần đầu")
            check(cache.stats.hits > 0, f"cache hits = {cache.stats.hits}")

        # ----------------------------------------------------- 5. retry ----
        print("\n[5] Retry — 503 hai lần đầu")
        script = ServerScript(fail_times=2)
        with StubServer(script) as url:
            cache = DiskCache(root / "c5", enabled=False)
            q = build_qwen_runner(f"{url}/v1", "qwen-stub", image_fn,
                                  model_name=TARGET_MODEL, api_key="stub", cache=cache)
            q.client.http.retry.base_delay = 0.05      # test nhanh
            pr = q("p0.png")
            check(len(pr.boxes) == 4, "vượt qua 2 lần 503, kết quả vẫn đúng")
            check(q.client.http.stats.retries == 2,
                  f"đếm đúng {q.client.http.stats.retries} lần retry")

        # ----------------------------------------- 6. output hỏng -> INVALID
        print("\n[6] Output không parse được => INVALID, KHÔNG phải Easy rỗng")
        script = ServerScript(unparseable=True)
        with StubServer(script) as url:
            cache = DiskCache(root / "c6", enabled=False)
            runners = build_runners(url, image_fn, cache)
            rec = run_cmcv_page(CMCV(runners, CMCVConfig()), "p0.png")
            check(all(t == Tier.INVALID for t in rec.tier.values()),
                  f"mọi subtask INVALID: {dict((k, v.value) for k, v in rec.tier.items())}")
            check(not any(t == Tier.EASY for t in rec.tier.values()),
                  "KHÔNG có subtask nào lọt vào EASY")
            check(len(runners[TARGET_MODEL].quarantine) == 1,
                  f"trang bị cách ly: {list(runners[TARGET_MODEL].quarantine)}")

        print("\n[6b] Block rỗng (trang trắng) cũng không được thành Easy rỗng")
        script = ServerScript(empty_blocks=True)
        with StubServer(script) as url:
            cache = DiskCache(root / "c6b", enabled=False)
            runners = build_runners(url, image_fn, cache)
            rec = run_cmcv_page(CMCV(runners, CMCVConfig()), "p0.png")
            # Qwen trả [] hợp lệ; Paddle/Mistral coi 'không block nào' là lỗi shape
            check(all(t == Tier.INVALID for t in rec.tier.values()),
                  f"INVALID thay vì EASY rỗng: {dict((k, v.value) for k, v in rec.tier.items())}")

        # ------------------------------------------------- 7. judge §3.3 ---
        print("\n[7] Judge §3.3 qua HTTP thật")
        from ..judge_refine import make_judge_fn
        from ..clients.openai_compat import OpenAICompatClient, OpenAICompatConfig
        script = ServerScript()
        with StubServer(script) as url:
            c = OpenAICompatClient(
                OpenAICompatConfig(model="gpt-5-stub", base_url=f"{url}/v1",
                                   api_key="stub", json_mode=False),
                DiskCache(root / "c7", enabled=False))
            judge_fn = make_judge_fn(c.as_call_model("judge-v1"))
            img = image_fn("p0.png")
            v = judge_fn(img, img, "E = mc^2", "formula")
            check(v.has_error and v.corrected == "E = mc^{2}",
                  f"verdict: has_error={v.has_error} corrected={v.corrected!r}")
            check(v.confidence == 0.82 and "hàng 2" in v.note,
                  f"confidence={v.confidence} note={v.note!r}")

        n_ok = sum(_results)
        print(f"\n{'='*64}\n{n_ok}/{len(_results)} kiểm tra ĐẠT")
        return 0 if n_ok == len(_results) else 1
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
