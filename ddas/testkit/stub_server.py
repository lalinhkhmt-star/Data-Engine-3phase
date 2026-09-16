"""Stub HTTP cho 4 nhà cung cấp — kiểm chứng lớp client KHÔNG cần mạng/tiền.

Không phải mock ở mức Python: server này nói HTTP thật trên localhost, nên nó
kiểm được đúng những thứ mock hay bỏ sót — mã lỗi, retry/backoff, circuit
breaker, timeout, và hình dạng JSON thật của từng nhà cung cấp.

Kịch bản lỗi bật được qua `ServerScript` để test các nhánh hỏng:
  - `fail_times`    : trả 503 n lần đầu rồi mới OK  -> kiểm retry
  - `unparseable`   : trả text không phải JSON       -> kiểm SafeRunner cách ly
  - `empty_blocks`  : trả blocks rỗng                 -> kiểm KHÔNG bị coi là Easy rỗng
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional


@dataclass
class ServerScript:
    """Điều khiển hành vi stub cho từng bài test."""
    fail_times: int = 0                 # số lần đầu trả 503
    unparseable: bool = False           # VLM trả chữ không phải JSON
    empty_blocks: bool = False          # MỌI model trả rỗng (trang trắng / lỗi shape)
    # Nội dung mỗi model "đọc" được — đặt khác nhau để ép ra tier mong muốn.
    blocks: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    calls: Dict[str, int] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def hit(self, name: str) -> int:
        with self._lock:
            self.calls[name] = self.calls.get(name, 0) + 1
            return self.calls[name]


DEFAULT_BLOCKS = [
    {"bbox": [100, 60, 900, 130], "type": "title", "content": "BÁO CÁO TÀI CHÍNH QUÝ III"},
    {"bbox": [100, 160, 900, 420], "type": "text",
     "content": "Doanh thu thuần tăng trưởng so với cùng kỳ năm trước."},
    {"bbox": [100, 450, 900, 700], "type": "table",
     "content": "<table><tr><td>Chỉ tiêu</td><td>Giá trị</td></tr>"
                "<tr><td>Doanh thu</td><td>1.250</td></tr></table>"},
    {"bbox": [100, 730, 900, 800], "type": "formula", "content": "E = mc^2"},
]


class _Handler(BaseHTTPRequestHandler):
    script: ServerScript = ServerScript()

    def log_message(self, *a):          # im lặng, đừng rác output test
        pass

    def _send(self, code: int, obj: Any) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:          # noqa: N802
        n = int(self.headers.get("Content-Length") or 0)
        try:
            req = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            req = {}
        path = self.path.split("?")[0]
        s = _Handler.script

        if s.fail_times and s.hit("fail") <= s.fail_times:
            self._send(503, {"error": "stub: quá tải tạm thời"})
            return

        if path.endswith("/chat/completions"):
            self._send(200, self._openai(req, s))
        elif path.endswith("/ocr"):
            self._send(200, self._mistral(s))
        elif "layout-parsing" in path:
            self._send(200, self._paddle(s))
        elif "generateContent" in path:
            self._send(200, self._gemini(s))
        else:
            self._send(404, {"error": f"stub: không có route {path}"})

    # ------------------------------------------------------------ providers --
    def _openai(self, req: Dict[str, Any], s: ServerScript) -> Dict[str, Any]:
        model = req.get("model", "?")
        s.hit(f"openai:{model}")
        sys_prompt = ""
        for m in req.get("messages") or []:
            if m.get("role") == "system":
                sys_prompt = m.get("content") or ""

        if "kiểm tra chất lượng" in sys_prompt:      # vai trọng tài §3.3
            text = json.dumps({"differences": ["thiếu dấu & ở hàng 2"],
                               "has_error": True, "confidence": 0.82,
                               "corrected": "E = mc^{2}",
                               "note": "hàng 2, cột 1: thiếu ký hiệu căn"},
                              ensure_ascii=False)
        elif s.unparseable:
            text = "Xin lỗi, tôi không thể xử lý ảnh này."
        elif s.empty_blocks:
            text = json.dumps({"blocks": []})
        else:
            blocks = s.blocks.get(model, DEFAULT_BLOCKS)
            text = "```json\n" + json.dumps({"blocks": blocks}, ensure_ascii=False) + "\n```"
        return {"choices": [{"message": {"role": "assistant", "content": text}}],
                "usage": {"prompt_tokens": 1600, "completion_tokens": 700}}

    def _mistral(self, s: ServerScript) -> Dict[str, Any]:
        s.hit("mistral")
        blocks = [] if s.empty_blocks else s.blocks.get("mistral-ocr-4", DEFAULT_BLOCKS)
        return {"pages": [{
            "index": 0,
            "dimensions": {"width": 1000, "height": 1000},
            "blocks": [{"bbox": b["bbox"], "type": b["type"], "markdown": b["content"]}
                       for b in blocks],
        }]}

    def _paddle(self, s: ServerScript) -> Dict[str, Any]:
        s.hit("paddle")
        blocks = [] if s.empty_blocks else s.blocks.get("paddleocr-vl", DEFAULT_BLOCKS)
        return {"result": {"layoutParsingResults": [{"prunedResult": {
            "parsing_res_list": [{"block_bbox": b["bbox"], "block_label": b["type"],
                                  "block_content": b["content"]} for b in blocks]}}]}}

    def _gemini(self, s: ServerScript) -> Dict[str, Any]:
        s.hit("gemini")
        return {"candidates": [{"content": {"parts": [
            {"text": "E = mc^{2}"}]}}],
            "usageMetadata": {"promptTokenCount": 1500, "candidatesTokenCount": 40}}


class StubServer:
    """Context manager: `with StubServer(script) as url: ...`"""

    def __init__(self, script: Optional[ServerScript] = None, host: str = "127.0.0.1"):
        self.script = script or ServerScript()
        _Handler.script = self.script
        self.httpd = ThreadingHTTPServer((host, 0), _Handler)
        self.port = self.httpd.server_address[1]
        self.url = f"http://{host}:{self.port}"
        self._t: Optional[threading.Thread] = None

    def __enter__(self) -> str:
        self._t = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self._t.start()
        return self.url

    def __exit__(self, *exc) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        if self._t:
            self._t.join(timeout=5)
