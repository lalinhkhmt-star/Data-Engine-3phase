"""Ghi/đọc kết quả pipeline ra đĩa.

KHÔNG có trong paper, nhưng thiếu thì không chạy thật được: pipeline.py:3 mô tả
"mỗi bước là một job riêng, checkpoint ra parquet" mà không có dòng code nào
làm — chạy xong là mất sạch, và mọi bước sau không tiếp tục được từ giữa chừng.

Định dạng: JSONL (mỗi dòng một bản ghi). pandas/pyarrow CÓ sẵn trong môi trường
nên parquet là lựa chọn khả thi, nhưng vẫn chọn JSONL ở quy mô hiện tại vì:
  - nối thêm được từng dòng, nên job dài chết giữa chừng vẫn giữ phần đã chạy
    (parquet phải ghi trọn row-group mới đọc lại được),
  - đọc bằng mắt/grep/jq được khi đi soi lỗi nhãn — việc sẽ làm rất nhiều.
Khi lên hàng chục triệu dòng thì đổi sang parquet qua đúng hai hàm
`write_jsonl`/`read_jsonl` bên dưới, phần còn lại không phải sửa.

numpy array (bbox) được chuyển thành list khi ghi và dựng lại khi đọc, vì json
không hiểu ndarray.
"""
from __future__ import annotations

import gzip
import json
import os
from dataclasses import asdict, fields, is_dataclass
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Type

import numpy as np

from .cmcv import Tier


def _encode(v: Any) -> Any:
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, Tier):
        return v.value
    if isinstance(v, dict):
        return {k: _encode(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_encode(x) for x in v]
    if is_dataclass(v):
        return {k: _encode(x) for k, x in asdict(v).items()}
    return v


def _open(path: str, mode: str):
    return gzip.open(path, mode + "t", encoding="utf-8") if path.endswith(".gz") \
        else open(path, mode, encoding="utf-8")


def write_jsonl(path: str, records: Iterable[Any], append: bool = False) -> int:
    """Ghi các dataclass/dict ra JSONL. `.gz` ở đuôi tên thì tự nén.

    `append=True` để job chạy theo lô nối tiếp vào cùng file — cần khi chạy
    thật trên hàng chục triệu trang, không giữ hết trong RAM được.
    """
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    n = 0
    with _open(path, "a" if append else "w") as fh:
        for r in records:
            fh.write(json.dumps(_encode(r), ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    """Đọc lười từng dòng — file lớn không nạp hết vào RAM."""
    with _open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


_ARRAY_FIELDS = {"box", "boxes"}
_TIER_FIELDS = {"tier"}


def load_dataclass(path: str, cls: Type) -> List[Any]:
    """Đọc JSONL dựng lại thành dataclass `cls`, khôi phục ndarray và Tier.

    Bỏ qua khoá lạ (file ghi bởi phiên bản cũ/mới hơn) thay vì nổ — dữ liệu đã
    chạy tốn tiền rồi, không nên mất chỉ vì thêm một trường.
    """
    names = {f.name for f in fields(cls)}
    out: List[Any] = []
    for d in read_jsonl(path):
        kw: Dict[str, Any] = {}
        for k, v in d.items():
            if k not in names:
                continue
            if k in _ARRAY_FIELDS and isinstance(v, list):
                v = np.asarray(v, dtype=np.float32)
            elif k in _TIER_FIELDS and isinstance(v, str):
                v = Tier(v)
            kw[k] = v
        out.append(cls(**kw))
    return out


# ------------------------------------------------- xuất trọn bộ Data Engine --

# Tên file cố định để các bước sau tìm được mà không phải truyền tham số lung tung.
LAYOUT = {
    "stage1_pretrain": "stage1_pretrain.jsonl",    # Easy+Medium, nhãn đồng thuận CMCV
    "stage2_sft": "stage2_sft.jsonl",              # Hard đã chú thích tay
    "stage3_grpo": "stage3_grpo.jsonl",            # Hard chấm tự động được
    "expert_queue": "expert_queue.jsonl",          # hàng đợi người, đã xếp ưu tiên
    "refined": "judge_refined.jsonl",              # Hard tầng tự động cứu được
    "scanqa": "scanqa.jsonl",                      # số đo chất lượng ảnh
    "manifest": "manifest.json",
}


def export_dataset(out_dir: str,
                   stage1: Optional[Iterable[Any]] = None,
                   stage2: Optional[Iterable[Any]] = None,
                   stage3: Optional[Iterable[Any]] = None,
                   expert_queue: Optional[Iterable[Any]] = None,
                   refined: Optional[Iterable[Any]] = None,
                   scanqa: Optional[Iterable[Any]] = None,
                   extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Xuất bộ dữ liệu phân tầng theo đúng dòng 66 paper + manifest đếm số.

    Manifest là thứ đọc đầu tiên khi quay lại sau vài tuần: mỗi tầng bao nhiêu
    mẫu, chia theo subtask thế nào. Không có nó thì phải đếm lại bằng tay.
    """
    counts: Dict[str, Any] = {}
    for key, data in (("stage1_pretrain", stage1), ("stage2_sft", stage2),
                      ("stage3_grpo", stage3), ("expert_queue", expert_queue),
                      ("refined", refined), ("scanqa", scanqa)):
        if data is None:
            continue
        rows = list(data)
        path = os.path.join(out_dir, LAYOUT[key])
        write_jsonl(path, rows)
        by_st: Dict[str, int] = {}
        for r in rows:
            st = getattr(r, "subtask", None)
            if st:
                by_st[st] = by_st.get(st, 0) + 1
        counts[key] = {"n": len(rows), "file": LAYOUT[key], **({"theo_subtask": by_st} if by_st else {})}

    manifest = {"tầng": counts, **(extra or {})}
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, LAYOUT["manifest"]), "w", encoding="utf-8") as fh:
        json.dump(_encode(manifest), fh, ensure_ascii=False, indent=2)
    return manifest
