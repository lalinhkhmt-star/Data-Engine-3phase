"""Stage 2 — Lấy mẫu mức element. Layout detection CHẠY TRƯỚC, CMCV chạy SAU.

Đúng thứ tự trong Figure 3 (paper): Page Data -> Layout Detection -> Text/
Formula/Table (bbox+class) -> cluster riêng từng loại -> CMCV so sánh nội
dung -> Final Sample. Bbox+class KHÔNG suy ra từ việc match 2 model CMCV với
nhau (cách cũ, sai thứ tự) — mà từ 1 model layout detection ĐỘC LẬP
(HeronLayoutDetector, xem layout_heron.py) chạy trên ảnh trang, tách biệt
hoàn toàn khỏi 3 model CMCV.

Với mỗi bbox Heron phát hiện, tra nội dung tương ứng (IoU tốt nhất, cùng
nhãn) trong output ĐàCÓ SẴN của target/cheap/expensive — KHÔNG suy luận
model CMCV thêm lần nào (chỉ Heron là suy luận thêm thật sự, xem chi phí
trong costmodel.py). Model nào không có nội dung khớp tại vùng đó (bbox
Heron thấy nhưng model CMCV bỏ sót) -> không đủ bằng chứng so sánh -> Hard,
không đoán.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from .cluster import ClusterIndex, hierarchical_cluster
from .cmcv import CHEAP_EXTERNAL, TARGET_MODEL, ParseResult, Tier, assign_tier
from .config import ClusterConfig, CMCVConfig, SamplerConfig
from .layout_heron import LayoutBox
from .metrics import SIM_FN, iou_matrix
from .sampler import Allocation, allocate_nested, draw


@dataclass
class Element:
    page_id: str
    eid: int
    etype: str                 # 'text' | 'formula' | 'table'
    box: np.ndarray            # xyxy trên trang gốc
    content: Dict[str, str]    # model -> nội dung
    tier: Tier
    sims: Dict[str, Optional[float]]


# Nhãn block -> subtask huấn luyện. Tiêu đề/mục danh sách/chú thích LÀ text,
# nên chúng phải rơi vào subtask 'text'. Trước đây phép ghép so nhãn đã-gộp
# ("text") với nhãn anchor chưa gộp ("title") nên KHÔNG BAO GIỜ khớp, khiến
# mọi element title/list/caption bị gán Hard rồi rơi khỏi cả 3 subtask — bị
# loại âm thầm khỏi Stage 2. Giữ nhãn nguyên vẹn khi ghép, gộp về subtask SAU.
SUBTASK_OF_LABEL = {
    "text": "text", "title": "text", "list": "text", "caption": "text",
    "formula": "formula", "table": "table",
}


def _elements_of(r: ParseResult) -> List[Tuple[str, np.ndarray, Optional[str]]]:
    """Làm phẳng ParseResult thành (nhãn, box, nội dung) theo thứ tự đọc.

    Nội dung `None` nghĩa là KHÔNG BIẾT — model này không cung cấp được nội
    dung theo block tại vùng đó. Phân biệt None với chuỗi rỗng là bắt buộc:
    chuỗi rỗng là một khẳng định ("vùng này không có chữ") và hai khẳng định
    rỗng sẽ "đồng thuận" với nhau, còn None thì derive_element_cmcv gán Hard vì
    thiếu bằng chứng. Gộp hai thứ này chính là lỗi đã sinh ra nhãn EASY RỖNG
    cho toàn bộ subtask text.
    """
    out: List[Tuple[str, np.ndarray, Optional[str]]] = []
    has_contents = len(r.contents) == len(r.labels)
    fi = ti = 0
    for i, (box, lab) in enumerate(zip(r.boxes, r.labels)):
        if has_contents:
            content: Optional[str] = r.contents[i]
        elif lab == "formula":
            content = r.formulas[fi] if fi < len(r.formulas) else None
        elif lab == "table":
            content = r.tables[ti] if ti < len(r.tables) else None
        else:
            # Adapter cũ không có `contents`: nội dung text từng block KHÔNG
            # khôi phục được từ chuỗi ghép r.text -> không biết, chứ không rỗng.
            content = None
        if lab == "formula":
            fi += 1
        elif lab == "table":
            ti += 1
        if lab in SUBTASK_OF_LABEL:
            out.append((lab, box, content))
    return out


def _match_content_to_boxes(anchor: Sequence[LayoutBox], r: ParseResult,
                            iou_thr: float = 0.5) -> List[Optional[str]]:
    """Với mỗi bbox neo (từ layout detector độc lập), tra nội dung tương ứng
    trong ParseResult của 1 model CMCV (IoU tốt nhất, cùng nhãn). None nếu
    model đó không có gì khớp tại vùng đó (bbox Heron thấy nhưng model bỏ sót).
    """
    elems = _elements_of(r)
    if not anchor:
        return []
    if not elems:
        return [None] * len(anchor)
    A = np.stack([a.box for a in anchor])
    B = np.stack([e[1] for e in elems])
    M = iou_matrix(A, B)
    same = np.array([[a.label == e[0] for e in elems] for a in anchor])
    M = np.where(same, M, 0.0)
    out: List[Optional[str]] = []
    for i in range(len(anchor)):
        j = int(np.argmax(M[i]))
        out.append(elems[j][2] if M[i, j] >= iou_thr else None)
    return out


def derive_element_cmcv(page_id: str, layout_boxes: Sequence[LayoutBox],
                        rm: ParseResult, rp: ParseResult, rq: Optional[ParseResult],
                        cfg: CMCVConfig | None = None) -> List[Element]:
    """CMCV mức element, ĐI SAU layout detection (đúng Figure 3 paper).

    `layout_boxes` là bbox+class từ HeronLayoutDetector — nguồn "có gì trên
    trang" độc lập với 3 model CMCV. Với mỗi bbox, tra nội dung tương ứng
    của target/cheap/expensive rồi so sánh — tái sử dụng taxonomy
    Easy/Medium/Hard của trang, không suy luận model CMCV thêm lần nào.
    """
    cfg = cfg or CMCVConfig()
    ca_list = _match_content_to_boxes(layout_boxes, rm)
    cb_list = _match_content_to_boxes(layout_boxes, rp)
    cq_list = _match_content_to_boxes(layout_boxes, rq) if rq is not None else [None] * len(layout_boxes)

    out: List[Element] = []
    for k, lb in enumerate(layout_boxes):
        # `lb.label` là nhãn block (text/title/list/caption/formula/table);
        # `etype` là SUBTASK huấn luyện mà element này thuộc về.
        etype = SUBTASK_OF_LABEL.get(lb.label, "text")
        ca, cb, cq = ca_list[k], cb_list[k], cq_list[k]
        content = {m: c for m, c in ((TARGET_MODEL, ca), (CHEAP_EXTERNAL, cb)) if c is not None}
        no_evidence = (ca is None or cb is None)
        if not no_evidence and not ca.strip() and not cb.strip():
            # CẢ HAI cùng rỗng. Về mặt số học đây là "đồng thuận hoàn hảo"
            # (mọi sim = 1.0) và sẽ thành EASY với nhãn rỗng — đúng cái bẫy đã
            # bơm nhãn rỗng vào 42% dataset. Không có nội dung thì không có gì
            # để huấn luyện, nên coi là thiếu bằng chứng. Một bên rỗng một bên
            # có chữ thì KHÔNG rơi vào đây: đó là bất đồng thật, cứ so bình
            # thường rồi để điểm tương đồng thấp tự đẩy xuống Hard.
            no_evidence = True
        if no_evidence:
            # target hoặc cheap-external không có nội dung khớp bbox này ->
            # không đủ bằng chứng so sánh -> không đoán, gán thẳng Hard.
            out.append(Element(page_id, k, etype, lb.box, content, Tier.HARD,
                               {"M-P": None, "M-Q": None, "P-Q": None}))
            continue
        fn = SIM_FN[etype] if etype in SIM_FN else SIM_FN["text"]
        tau = cfg.tau[etype]
        s_mp = fn(ca, cb)
        s_mq = fn(ca, cq) if cq is not None else None
        s_pq = fn(cb, cq) if cq is not None else None
        tier = assign_tier(s_mp, s_mq, s_pq, tau, cfg.require_3way_for_easy)
        out.append(Element(page_id, k, etype, lb.box, content, tier,
                           {"M-P": s_mp, "M-Q": s_mq, "P-Q": s_pq}))
    return out


# --------------------------------------------------- đặc trưng element ------

def element_feature(crop_embed: np.ndarray, box: np.ndarray,
                    page_wh: Tuple[float, float], etype: str,
                    extra: Optional[np.ndarray] = None) -> np.ndarray:
    """Đặc trưng cụm mức element: hình ảnh crop + tiên nghiệm hình học.

    Tiên nghiệm hình học (tỉ lệ khung, diện tích tương đối, vị trí) rẻ nhưng
    tách rất tốt: bảng rộng-thấp vs bảng lồng nhau, công thức inline vs display.
    """
    w, h = page_wh
    x1, y1, x2, y2 = box
    bw, bh = max(x2 - x1, 1e-6), max(y2 - y1, 1e-6)
    geo = np.array([bw / w, bh / h, (bw * bh) / (w * h), bw / bh,
                    (x1 + x2) / (2 * w), (y1 + y2) / (2 * h)], np.float32)
    parts = [crop_embed.astype(np.float32), geo]
    if extra is not None:
        parts.append(extra.astype(np.float32))
    v = np.concatenate(parts)
    return v / max(np.linalg.norm(v), 1e-9)


# --------------------------------------- pipeline mức element (Stage 2) -----
#
# Khác Stage 1 (trang): KHÔNG cần probe-and-extrapolate. Ở stage 1, probe tồn
# tại để né chạy CMCV trên toàn pool 500M trang. Ở đây, element chỉ được tạo
# ra từ candidate set trang đã có CMCV đầy đủ (stage 1), nên tier của MỌI
# element đã biết CHÍNH XÁC (dẫn xuất, không suy luận thêm — xem đầu file) —
# đi thẳng vào cluster + allocate_nested, không cần ước lượng.

def crop_box(image: Image.Image, box: np.ndarray) -> Image.Image:
    """Cắt vùng bbox ra khỏi ảnh trang gốc, chặn biên để không vượt kích thước ảnh."""
    x1, y1, x2, y2 = [float(v) for v in box]
    W, H = image.size
    xi1, yi1 = max(int(x1), 0), max(int(y1), 0)
    xi2 = min(max(int(x2), xi1 + 1), W)
    yi2 = min(max(int(y2), yi1 + 1), H)
    return image.crop((xi1, yi1, xi2, yi2))


def build_elements(page_ids: Sequence[str],
                   layout_fn: Callable[[str], Sequence[LayoutBox]],
                   parse_fn: Callable[[str], Tuple[ParseResult, ParseResult, Optional[ParseResult]]],
                   cfg: CMCVConfig | None = None) -> List[Element]:
    """Chạy layout detection rồi derive_element_cmcv trên tập trang candidate.

    `layout_fn(page_id)` -> bbox+class từ HeronLayoutDetector (suy luận THẬT,
    xem chi phí trong costmodel.py). `parse_fn(page_id)` PHẢI trả lại đúng
    ParseResult đã tính ở CMCV trang (cache lại, không gọi model/API lần nữa).
    """
    out: List[Element] = []
    for pid in page_ids:
        boxes = layout_fn(pid)
        rm, rp, rq = parse_fn(pid)
        out.extend(derive_element_cmcv(pid, boxes, rm, rp, rq, cfg))
    return out


def embed_elements(elements: Sequence[Element],
                   image_fn: Callable[[str], Image.Image],
                   encoder, batch_size: int = 512) -> List[np.ndarray]:
    """Cắt từng element theo bbox trên ảnh trang gốc rồi embed theo lô.

    `encoder` là bất kỳ object có `.encode(list[PIL.Image]) -> (N,d) ndarray`
    (vd. embed_real.ViTPageEncoder, nên dùng model NHỎ hơn model nhúng trang —
    xem RATE['vit_small_crop'] trong costmodel.py, ~3.5x nhanh hơn vit_base).
    Cache ảnh trang theo page_id để không render/tải lại nhiều lần cho các
    element cùng trang.
    """
    cache: Dict[str, Image.Image] = {}
    crops = []
    for el in elements:
        img = cache.get(el.page_id)
        if img is None:
            img = image_fn(el.page_id)
            cache[el.page_id] = img
        crops.append(crop_box(img, el.box))
    out: List[np.ndarray] = []
    for s in range(0, len(crops), batch_size):
        out.append(encoder.encode(crops[s:s + batch_size]))
    return list(np.concatenate(out, axis=0)) if out else []


def cluster_and_sample(elements: Sequence[Element], embeds: Sequence[np.ndarray],
                       page_wh: Dict[str, Tuple[float, float]], subtask: str,
                       cluster_cfg: ClusterConfig, sampler_cfg: SamplerConfig,
                       gains: Dict[str, Tuple[float, float, float]],
                       budget: int, seed: int = 0
                       ) -> Tuple[List[int], Optional[ClusterIndex], Optional[Allocation]]:
    """Cluster riêng cho element thuộc `subtask` rồi lấy mẫu lồng nhau (cụm x độ khó).

    Trả về (chỉ số vào `elements` của các element được chọn, ClusterIndex, Allocation).
    """
    idx = [i for i, e in enumerate(elements) if e.etype == subtask]
    if not idx:
        return [], None, None
    feats = np.stack([element_feature(embeds[i], elements[i].box,
                                      page_wh[elements[i].page_id], subtask)
                      for i in idx])
    ci = hierarchical_cluster(feats, cluster_cfg, seed=seed)

    avail: Dict = defaultdict(int)
    members: Dict = defaultdict(list)
    for local_i, gi in enumerate(idx):
        cell = (int(ci.assign[local_i]), elements[gi].tier)
        avail[cell] += 1
        members[cell].append(gi)

    alloc = allocate_nested(dict(avail), subtask, sampler_cfg, gains, ci.paths, budget=budget)
    chosen = draw(alloc, members, seed=seed + 1)
    return chosen, ci, alloc
