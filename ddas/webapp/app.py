"""Web UI để test pipeline: đẩy 1 ảnh vào, xem từng giai đoạn + log real-time.

Chạy:
    uvicorn ddas.webapp.app:app --reload --port 8000
    mở http://localhost:8000

Đây là công cụ TEST/DEV, không phải dịch vụ production: không auth, chạy
1 process, state (hàng đợi sự kiện mỗi lần chạy) giữ trong RAM. Dùng để soát
bằng mắt pipeline thật (§3.1 mức trang + §3.2 CMCV + §3.3 judge/preannot) trên
model thật, không phải để phục vụ nhiều người dùng đồng thời.
"""
from __future__ import annotations

import logging
import queue
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# Nạp .env NGAY ĐẦU module — trước khi bất kỳ chỗ nào đọc os.environ. Biến đã
# export sẵn trong shell luôn thắng (python-dotenv mặc định override=False),
# .env chỉ điền chỗ trống. Không có .env thì im lặng bỏ qua (find_dotenv()
# trả rỗng), không lỗi.
from dotenv import load_dotenv as _load_dotenv
_load_dotenv()

from ..clients import EngineClients, build_clients, env_report
from ..config import DDASConfig
from .runner import run_pipeline

log = logging.getLogger("ddas.webapp")

APP_DIR = Path(__file__).parent
UPLOAD_DIR = APP_DIR / "uploads"       # gitignored — ảnh người dùng tải lên mỗi lần chạy
UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI(title="DDAS pipeline — test 1 ảnh")
app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")

# run_id -> queue.Queue[dict | None]  (None = kết thúc luồng SSE)
_RUNS: Dict[str, "queue.Queue"] = {}
_RUNS_LOCK = threading.Lock()
_RUN_TTL_S = 3600.0          # dọn run cũ sau 1 giờ để không phình RAM


def _cleanup_old_runs() -> None:
    now = time.time()
    with _RUNS_LOCK:
        stale = [rid for rid, q in _RUNS.items()
                if getattr(q, "_created", now) < now - _RUN_TTL_S]
        for rid in stale:
            _RUNS.pop(rid, None)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(APP_DIR / "static" / "index.html"))


@app.get("/api/env")
def api_env() -> JSONResponse:
    """Vai nào đã cấu hình — frontend hiển thị TRƯỚC khi người dùng bấm chạy,
    để biết trước giai đoạn nào sẽ bị bỏ qua thay vì phải chạy mới biết."""
    return JSONResponse(env_report())


@app.post("/api/run")
async def api_run(
    image: UploadFile = File(...),
    subtasks: str = Form("text,formula,table"),
    force_scanqa: bool = Form(False),
    heron_device: str = Form("cpu"),
) -> JSONResponse:
    _cleanup_old_runs()
    run_id = uuid.uuid4().hex[:12]
    run_dir = UPLOAD_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(image.filename or "page.png").suffix or ".png"
    dest = run_dir / f"page{suffix}"
    with dest.open("wb") as f:
        shutil.copyfileobj(image.file, f)

    q: "queue.Queue" = queue.Queue()
    q._created = time.time()          # type: ignore[attr-defined]
    with _RUNS_LOCK:
        _RUNS[run_id] = q

    subtask_list = tuple(s.strip() for s in subtasks.split(",") if s.strip())

    def worker() -> None:
        try:
            clients: EngineClients = build_clients(pages_root=str(run_dir))
            cfg = DDASConfig()
            for ev in run_pipeline(dest.name, clients, cfg, subtasks=subtask_list,
                                   force_past_scanqa=force_scanqa,
                                   heron_device=heron_device):
                q.put(ev.to_dict())
        except Exception as e:                     # noqa: BLE001 — phải bắt hết, đây là luồng nền
            log.exception("run %s hỏng", run_id)
            q.put({"stage": "fatal", "status": "fatal",
                  "message": f"lỗi không bắt được: {type(e).__name__}: {e}", "data": {},
                  "ts": time.time()})
        finally:
            q.put(None)      # đóng SSE

    threading.Thread(target=worker, daemon=True).start()
    return JSONResponse({"run_id": run_id})


@app.get("/api/stream/{run_id}")
def api_stream(run_id: str) -> StreamingResponse:
    with _RUNS_LOCK:
        q = _RUNS.get(run_id)
    if q is None:
        # Không có 'event: end' thì EventSource phía trình duyệt coi kết nối là
        # bị rớt (không phải kết thúc bình thường) và TỰ ĐỘNG RECONNECT vô hạn
        # — nút "Chạy pipeline" kẹt ở trạng thái "Đang chạy…" mãi mãi. Phải
        # đóng luồng đúng cách giống nhánh chạy thật (xem gen() bên dưới).
        body = ('data: {"stage":"fatal","status":"fatal",'
               '"message":"run_id không tồn tại hoặc đã hết hạn"}\n\n'
               'event: end\ndata: {}\n\n')
        return StreamingResponse(iter([body]), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache"})

    def gen():
        import json
        while True:
            item = q.get()
            if item is None:
                yield "event: end\ndata: {}\n\n"
                break
            yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                     "X-Accel-Buffering": "no"})
