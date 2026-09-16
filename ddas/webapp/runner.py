"""Chạy pipeline THẬT trên 1 ảnh, phát StageEvent theo từng bước — dùng cho web UI.

Không sửa bất kỳ file nào trong ddas/*.py — module này chỉ GỌI core đã có
(cmcv, element, layout_heron, judge_refine, preannot, scanqa) và bọc thêm lớp
log sự kiện mỏng ở BÊN NGOÀI để phát ra SSE. Với 1 ảnh, các bước quy mô pool
(cluster/probe/expand) không áp dụng — bỏ qua, bắt đầu thẳng từ scanqa.

Thứ tự giai đoạn:
  ingest -> scanqa -> embed(thông tin) -> layout(Heron) -> cmcv(3 model)
  -> elements(derive) -> judge_refine(mẫu Hard) -> preannot(mẫu không tự cứu được)
  -> done(tổng kết)

Nguyên tắc quan trọng nhất được giữ nguyên từ pipeline thật (xem sft.py):
subtask 'layout' dùng tier MỨC TRANG (CMCVRecord.tier['layout']); subtask
'text'/'formula'/'table' dùng tier MỨC ELEMENT (derive_element_cmcv), KHÔNG
phải CMCVRecord.tier['text'] — hai thứ đó độc lập, page-level chỉ mang tính
thông tin cho ba subtask này.
"""
from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence

from PIL import Image

from ..clients import ClientError, EngineClients, run_cmcv_page
from ..cmcv import CHEAP_EXTERNAL, CMCV, EXPENSIVE_EXTERNAL, TARGET_MODEL, Tier
from ..config import DDASConfig
from ..element import derive_element_cmcv
from ..judge_refine import ExpertItem, JudgeRefine, RefinedRecord, make_judge_fn
from ..preannot import build_expert_task
from ..scanqa import assess as scanqa_assess
from ..sft import HardItem

Emit = Callable[["StageEvent"], None]


@dataclass
class StageEvent:
    stage: str            # ingest|scanqa|embed|layout|cmcv|elements|judge_refine|preannot|done|fatal
    status: str            # start|log|ok|warn|error|done|fatal
    message: str
    data: Optional[Dict[str, Any]] = None
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {"stage": self.stage, "status": self.status, "message": self.message,
                "data": self.data or {}, "ts": self.ts}


def _box_list(box) -> Optional[List[float]]:
    return None if box is None else [round(float(v), 1) for v in box]


def _preview(s: Optional[str], n: int = 220) -> str:
    if not s:
        return ""
    s = s.strip()
    return s if len(s) <= n else s[:n] + "…"


def run_pipeline(page_id: str, clients: EngineClients, cfg: Optional[DDASConfig] = None,
                 subtasks: Sequence[str] = ("text", "formula", "table"),
                 force_past_scanqa: bool = False,
                 heron_device: str = "cpu") -> Iterator[StageEvent]:
    """Sinh StageEvent lần lượt. Lỗi ở MỘT giai đoạn không dừng toàn bộ, trừ khi
    fatal (ảnh không đọc được) — mỗi giai đoạn tự quyết định có đi tiếp hay dừng.
    """
    cfg = cfg or DDASConfig()
    subtasks = [s for s in subtasks if s in ("text", "formula", "table")] or ["text"]
    t_start = time.monotonic()

    def E(stage: str, status: str, message: str, **data) -> StageEvent:
        return StageEvent(stage, status, message, data or None)

    # -------------------------------------------------------------- ingest --
    yield E("ingest", "start", f"nạp ảnh {page_id}")
    try:
        image: Image.Image = clients.image_fn(page_id)
    except Exception as e:
        yield E("ingest", "fatal", f"không đọc được ảnh: {e}")
        yield E("done", "fatal", "dừng — không có ảnh để xử lý")
        return
    W, H = image.size
    yield E("ingest", "ok", f"{W}x{H} px sau resize", width=W, height=H)

    # -------------------------------------------------------------- scanqa --
    yield E("scanqa", "start", "đo chất lượng ảnh (mờ/nghiêng/mực/tương phản)")
    qa = scanqa_assess(image, page_id)
    qa_data = dict(blur=round(qa.blur, 1), ink=round(qa.ink, 4),
                   contrast=round(qa.contrast, 1), skew_deg=round(qa.skew_deg, 2),
                   reasons=qa.reasons)
    if qa.drop:
        yield E("scanqa", "warn", "ảnh bị GẮN CỜ loại (INVALID): " + "; ".join(qa.reasons), **qa_data)
        if not force_past_scanqa:
            yield E("scanqa", "error", "dừng ở đây — production sẽ KHÔNG gọi model cho "
                                       "trang này (validity_fn). Tick 'bỏ qua cổng scanqa' để "
                                       "vẫn chạy tiếp cho mục đích test.")
            yield E("done", "warn", "dừng sớm vì scanqa — không tính là lỗi hệ thống")
            return
        yield E("scanqa", "log", "bỏ qua cổng (force) — chạy tiếp CHỈ để test, "
                                 "production sẽ không làm vậy")
    else:
        yield E("scanqa", "ok", "qua cổng chất lượng", **qa_data)

    # --------------------------------------------------------------- embed --
    # Thông tin tham khảo — với 1 ảnh không có pool để cluster nên không ảnh
    # hưởng các bước sau, chỉ cho thấy encoder có chạy được không.
    yield E("embed", "start", f"nhúng CLIP ({cfg.embed.vit_name})")
    try:
        from ..embed_real import ViTPageEncoder
        t0 = time.monotonic()
        enc = ViTPageEncoder(model_name=cfg.embed.vit_name, device="cpu")
        vec = enc.encode([image])[0]
        yield E("embed", "ok", f"vector {vec.shape[0]}-d, {time.monotonic()-t0:.1f}s",
               dim=int(vec.shape[0]), norm=round(float((vec ** 2).sum() ** 0.5), 3))
    except Exception as e:
        yield E("embed", "warn", f"bỏ qua (không bắt buộc cho 1 ảnh): {type(e).__name__}: {e}")

    # -------------------------------------------------------------- layout --
    layout_boxes = None
    yield E("layout", "start", "Docling Layout Heron — phát hiện bbox+class")
    try:
        from ..layout_heron import HeronLayoutDetector
        t0 = time.monotonic()
        det = HeronLayoutDetector(device=heron_device)
        layout_boxes = det.detect([image])[0]
        dt = time.monotonic() - t0
        yield E("layout", "ok", f"{len(layout_boxes)} vùng, {dt:.1f}s ({heron_device})",
               n_boxes=len(layout_boxes),
               boxes=[{"cls": b.cls, "score": round(b.score, 2), "box": _box_list(b.box)}
                      for b in layout_boxes[:60]])
    except Exception as e:
        yield E("layout", "warn", f"Heron không chạy được ({type(e).__name__}: {e}) — "
                                  "bước 'elements' (text/formula/table mức vùng) sẽ bị bỏ qua, "
                                  "CMCV mức trang vẫn chạy bình thường")

    # ---------------------------------------------------------------- cmcv --
    # cmcv.CMCV.run_page() truy cập runners[TARGET_MODEL] và runners[CHEAP_EXTERNAL]
    # KHÔNG ĐIỀU KIỆN (KeyError nếu thiếu — xem cmcv.py). EXPENSIVE_EXTERNAL thì
    # KHÁC: chỉ gọi khi cascade cần trọng tài, nên thiếu nó không crash ngay,
    # cascade chỉ mất tác dụng. Phải kiểm đúng 2 vai bắt buộc TRƯỚC khi vào CMCV,
    # nếu không lỗi rơi thành KeyError mù mờ, người dùng không biết thiếu key nào.
    missing_required = [m for m in (TARGET_MODEL, CHEAP_EXTERNAL) if m not in clients.runners]
    if missing_required:
        yield E("cmcv", "error",
               f"thiếu model BẮT BUỘC: {', '.join(missing_required)} — CMCV không chạy được "
               "nếu không có cả target lẫn cheap-external. Xem GET /api/env "
               "(cần QWEN_BASE_URL và MISTRAL_API_KEY tối thiểu).",
               missing=missing_required)
        yield E("done", "error", "dừng — thiếu cấu hình model bắt buộc")
        return
    runners_for_cmcv = dict(clients.runners)
    if EXPENSIVE_EXTERNAL not in runners_for_cmcv:
        # cmcv.CMCV.run_page() gọi runners[EXPENSIVE_EXTERNAL] KHÔNG ĐIỀU KIỆN
        # mỗi khi cascade cần leo thang (target/cheap bất đồng đủ mạnh) — nếu
        # thiếu key này thì KeyError bắn thẳng ra ngoài run_cmcv_page (hàm đó
        # chỉ bắt ClientError), sập cả generator. KHÔNG thể chỉ cảnh báo suông
        # rồi bỏ qua như 2 model bắt buộc — phải tiêm một runner giả LUÔN NÉM
        # ClientError khi bị gọi, để:
        #   (a) không có bất đồng cần leo thang -> runner giả không bao giờ bị
        #       gọi, pipeline chạy y hệt như đã test ở nhánh "thiếu Paddle,
        #       target~cheap đồng thuận" bên trên;
        #   (b) CÓ bất đồng cần leo thang -> ClientError bắn ra, rơi đúng vào
        #       đường run_cmcv_page() đã kiểm chứng (demo_clients.py bài [6]):
        #       cả trang thành INVALID, không suy đoán liều Hard/Easy.
        def _no_expensive(pid: str):
            raise ClientError(
                f"[{EXPENSIVE_EXTERNAL}] {pid}: PADDLE_VL_URL chưa cấu hình, "
                "cascade cần trọng tài nhưng không có")
        runners_for_cmcv[EXPENSIVE_EXTERNAL] = _no_expensive
        yield E("cmcv", "warn",
               f"thiếu {EXPENSIVE_EXTERNAL} (PADDLE_VL_URL) — nếu target/cheap bất đồng đủ "
               "mạnh để cascade cần trọng tài, cả trang sẽ thành INVALID (không suy đoán), "
               "không phải Hard")

    def instrumented(name: str, fn):
        def wrapped(pid: str):
            yield_box.append(E("cmcv", "log", f"gọi {name}…"))
            t0 = time.monotonic()
            try:
                pr = fn(pid)
            except ClientError as e:
                yield_box.append(E("cmcv", "warn", f"{name}: {e}"))
                raise
            dt = time.monotonic() - t0
            yield_box.append(E("cmcv", "ok",
                f"{name}: {dt:.1f}s · {len(pr.boxes)} box · {len(pr.text)} ký tự",
                model=name, latency_s=round(dt, 2), n_boxes=len(pr.boxes),
                text_preview=_preview(pr.text)))
            return pr
        return wrapped

    yield_box: List[StageEvent] = []
    yield E("cmcv", "start", f"CMCV cascade — {TARGET_MODEL} / {CHEAP_EXTERNAL} / {EXPENSIVE_EXTERNAL}")
    wrapped_runners = {name: instrumented(name, fn) for name, fn in runners_for_cmcv.items()}
    cmcv_obj = CMCV(wrapped_runners, cfg.cmcv)
    record = run_cmcv_page(cmcv_obj, page_id)
    yield from yield_box
    yield_box.clear()

    for st in ("layout", "text", "formula", "table"):
        sims = record.sims.get(st, {})
        yield E("cmcv", "ok" if record.tier[st] != Tier.HARD else "warn",
               f"[{st}] tier={record.tier[st].value}  nguồn={record.pseudo_label_from[st] or '-'}",
               subtask=st, tier=record.tier[st].value,
               sims={k: (round(v, 3) if v is not None else None) for k, v in sims.items()})
    yield E("cmcv", "done", f"used_expensive={record.used_expensive}",
           used_expensive=record.used_expensive)

    hard_items: List[HardItem] = []
    if record.tier["layout"] == Tier.HARD:
        hard_items.append(HardItem("layout", page_id, Tier.HARD))  # draft=None -> đi thẳng người

    # ------------------------------------------------------------ elements --
    if layout_boxes:
        yield E("elements", "start", f"derive mức vùng cho {', '.join(subtasks)}")
        try:
            rm = clients.runners[TARGET_MODEL](page_id) if TARGET_MODEL in clients.runners else None
            rp = clients.runners[CHEAP_EXTERNAL](page_id) if CHEAP_EXTERNAL in clients.runners else None
            rq = None
            if record.used_expensive and EXPENSIVE_EXTERNAL in clients.runners:
                rq = clients.runners[EXPENSIVE_EXTERNAL](page_id)
        except ClientError as e:
            yield E("elements", "warn", f"không lấy lại được ParseResult đã cache: {e} — bỏ qua")
            rm = rp = rq = None

        if rm is not None and rp is not None:
            elements = derive_element_cmcv(page_id, layout_boxes, rm, rp, rq, cfg.cmcv)
            elements = [e for e in elements if e.etype in subtasks]
            n_easy = sum(1 for e in elements if e.tier == Tier.EASY)
            n_medium = sum(1 for e in elements if e.tier == Tier.MEDIUM)
            n_hard = sum(1 for e in elements if e.tier == Tier.HARD)
            yield E("elements", "ok",
                   f"{len(elements)} vùng ({n_easy} Easy, {n_medium} Medium, {n_hard} Hard)",
                   n_elements=len(elements), easy=n_easy, medium=n_medium, hard=n_hard)
            for el in elements:
                content = el.content.get(TARGET_MODEL) or el.content.get(CHEAP_EXTERNAL) or ""
                yield E("elements",
                       "ok" if el.tier != Tier.HARD else "warn",
                       f"[{el.etype}] eid={el.eid} tier={el.tier.value}  {_preview(content, 100)!r}",
                       etype=el.etype, eid=el.eid, tier=el.tier.value, box=_box_list(el.box),
                       content_preview=_preview(content))
                if el.tier == Tier.HARD:
                    draft = el.content.get(TARGET_MODEL) or el.content.get(CHEAP_EXTERNAL) or None
                    hard_items.append(HardItem(el.etype, page_id, Tier.HARD, el.box, el.eid, draft))
        else:
            yield E("elements", "warn", "thiếu target/cheap ParseResult — bỏ qua")
    else:
        yield E("elements", "warn", "bỏ qua (không có bbox từ Heron)")

    # -------------------------------------------------------- judge_refine --
    refined: List[RefinedRecord] = []
    expert: List[ExpertItem] = []
    if not hard_items:
        yield E("judge_refine", "done", "không có mẫu Hard nào — không cần chạy")
    elif clients.judge_fn is None:
        yield E("judge_refine", "warn",
               f"{len(hard_items)} mẫu Hard nhưng THIẾU OPENAI_API_KEY — bỏ qua §3.3, "
               "mẫu Hard coi như đi thẳng hàng đợi người")
        expert = [ExpertItem(h.subtask, h.page_id, "no_judge_configured", 0.0, 0, h.draft,
                             "", "", h.box, h.element_id) for h in hard_items]
    else:
        yield E("judge_refine", "start",
               f"{len(hard_items)} mẫu Hard vào vòng render-then-verify (JUDGE_MODEL, tối đa "
               f"{cfg.judge.max_rounds} vòng)")

        current = {"page_id": "", "subtask": "", "element_id": None, "round": 0}

        def instrumented_judge(orig, shot, content, subtask):
            current["round"] += 1
            v = clients.judge_fn(orig, shot, content, subtask)
            yield_box.append(E("judge_refine",
                "ok" if not v.has_error else "log",
                f"[{current['subtask']}] eid={current['element_id']} vòng {current['round']}: "
                f"has_error={v.has_error} conf={v.confidence:.2f}"
                + (f"  note={_preview(v.note, 90)!r}" if v.note else ""),
                page_id=current["page_id"], subtask=current["subtask"],
                element_id=current["element_id"], round=current["round"],
                has_error=v.has_error, confidence=round(v.confidence, 2),
                note=v.note, corrected_preview=_preview(v.corrected)))
            return v

        jr = JudgeRefine(instrumented_judge, clients.image_fn, cfg.judge)
        for item in hard_items:
            current.update(page_id=item.page_id, subtask=item.subtask,
                           element_id=item.element_id, round=0)
            yield E("judge_refine", "start",
                   f"[{item.subtask}] eid={item.element_id}  nháp={_preview(item.draft, 80)!r}",
                   subtask=item.subtask, element_id=item.element_id)
            out = jr.run_item(item)
            yield from yield_box
            yield_box.clear()
            if isinstance(out, RefinedRecord):
                refined.append(out)
                yield E("judge_refine", "ok",
                       f"[{out.subtask}] eid={out.element_id} TỰ CỨU ĐƯỢC sau {out.rounds} vòng",
                       subtask=out.subtask, element_id=out.element_id, rounds=out.rounds,
                       content_preview=_preview(out.content))
            else:
                expert.append(out)
                yield E("judge_refine", "warn",
                       f"[{out.subtask}] eid={out.element_id} chuyển NGƯỜI — lý do: {out.reason}",
                       subtask=out.subtask, element_id=out.element_id, reason=out.reason,
                       located=out.located)
        yield E("judge_refine", "done",
               f"tự cứu {len(refined)}/{len(hard_items)} "
               f"({100*jr.resolve_rate:.0f}%) · {jr.stats['judge_calls']} lời gọi · "
               f"{jr.stats['render_failed']} render lỗi",
               resolved=len(refined), total=len(hard_items),
               resolve_rate=round(jr.resolve_rate, 3),
               judge_calls=jr.stats["judge_calls"], render_failed=jr.stats["render_failed"])

    # ------------------------------------------------------------ preannot --
    if expert:
        if clients.preannot_fn is None:
            yield E("preannot", "warn",
                   f"{len(expert)} mẫu chờ người nhưng THIẾU GEMINI_API_KEY — "
                   "bỏ qua pre-annotation, người sẽ phải gõ lại từ đầu")
        else:
            yield E("preannot", "start", f"pre-annotation độc lập cho {len(expert)} mẫu")
            for it in expert:
                try:
                    task = build_expert_task(it, clients.image_fn, clients.preannot_fn)
                except ClientError as e:
                    yield E("preannot", "warn", f"[{it.subtask}] eid={it.element_id}: {e}")
                    continue
                yield E("preannot", "ok",
                       f"[{task.subtask}] eid={task.element_id} luồng={task.mode} "
                       f"agreement={task.agreement if task.agreement is None else round(task.agreement,2)}",
                       subtask=task.subtask, element_id=task.element_id,
                       mode=task.mode,
                       agreement=None if task.agreement is None else round(task.agreement, 3),
                       draft_preview=_preview(task.draft), preannot_preview=_preview(task.preannot))
            yield E("preannot", "done", f"xong {len(expert)} mẫu")

    # ----------------------------------------------------------------- done --
    yield E("done", "done", f"hoàn tất trong {time.monotonic()-t_start:.1f}s",
           elapsed_s=round(time.monotonic() - t_start, 1),
           n_hard=len(hard_items), n_refined=len(refined), n_expert=len(expert),
           client_stats=clients.stats())
