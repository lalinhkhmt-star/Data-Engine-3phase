"""Prompt cho model trọng tài §3.3 (paper dòng 53 chỉ nói "judge-and-refine
prompt", KHÔNG cho nội dung — toàn bộ file này là thiết kế tự đề xuất).

Ba ràng buộc rút thẳng từ lập luận của chính paper, quyết định cách viết prompt:

1. Dòng 49 — "naive self-reflection has a systematic tendency to accept its own
   outputs". Nên prompt KHÔNG được hỏi trống "cái này đúng chưa?" (câu đó mời
   model trả lời "đúng rồi"). Phải bắt model LIỆT KÊ khác biệt thị giác cụ thể
   TRƯỚC, rồi mới kết luận — và nói thẳng tiên nghiệm: mẫu này đã bị 3 model
   CMCV bất đồng nên nhiều khả năng CÓ lỗi.
2. Dòng 51 — model yếu ở chiều "chuỗi -> hình dung ra ảnh". Nên prompt phải
   neo vào việc SO HAI ẢNH, không phải "đọc lại chuỗi LaTeX/HTML này".
3. Dòng 61 — tiêu chí ưu tiên #1 cần "lỗi đã được khoanh vùng" để người chỉ
   sửa cục bộ. Nên bắt buộc trường `note` mô tả lỗi Ở ĐÂU, không chỉ "có lỗi".

Prompt tách riêng theo subtask vì loại lỗi khác hẳn nhau: LaTeX hỏng cấu trúc
(thiếu &, \\\\ lệch hàng), HTML hỏng lưới (colspan sai, thiếu </td>), text thì
tiếng Việt chủ yếu sai dấu thanh/dấu phụ.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

# Schema đầu ra — khớp 1-1 với JudgeVerdict trong judge_refine.py.
VERDICT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["differences", "has_error", "confidence", "corrected", "note"],
    "properties": {
        "differences": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Liệt kê TRƯỚC mọi khác biệt thị giác quan sát được. Rỗng = không thấy khác biệt nào.",
        },
        "has_error": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0,
                       "description": "Độ chắc chắn của phán đoán has_error."},
        "corrected": {"type": ["string", "null"],
                      "description": "Bản sửa hoàn chỉnh. null nếu thấy lỗi nhưng không sửa nổi."},
        "note": {"type": "string",
                 "description": "Lỗi nằm Ở ĐÂU (hàng/cột/ký hiệu/vị trí trên trang) — người chú thích đọc cái này để sửa cục bộ."},
    },
}

JUDGE_SYSTEM = """\
Bạn là chuyên gia kiểm tra chất lượng bóc tách tài liệu. Bạn KHÔNG phải người \
tạo ra kết quả đang kiểm tra — nhiệm vụ của bạn là tìm lỗi trong kết quả của \
người khác.

Bối cảnh quan trọng: mẫu này đã được ba model bóc tách tài liệu độc lập xử lý \
và chúng BẤT ĐỒNG với nhau. Nghĩa là xác suất tiên nghiệm có lỗi là CAO. \
Kết luận "không có lỗi" chỉ được đưa ra sau khi đã soi và không tìm thấy khác \
biệt nào, không phải vì chưa soi kỹ.

Quy trình bắt buộc, theo đúng thứ tự:
1. Liệt kê vào `differences` MỌI khác biệt thị giác giữa hai ảnh — ký hiệu \
thiếu/thừa, sai vị trí, lệch hàng cột, chữ khác nhau, dấu thanh sai. Làm bước \
này TRƯỚC khi kết luận.
2. Chỉ khi `differences` rỗng mới được đặt `has_error = false`.
3. Nếu có lỗi: ghi `note` nói rõ lỗi NẰM Ở ĐÂU (hàng mấy, cột nào, ký hiệu \
nào, vị trí nào trên trang) — người chú thích sẽ dựa vào đó để sửa cục bộ.
4. Đưa bản sửa hoàn chỉnh vào `corrected`. Nếu không đủ căn cứ để sửa đúng, \
đặt `corrected = null` — KHÔNG đoán bừa, nhãn sai hại hơn không có nhãn.

Chỉ trả về JSON hợp lệ theo schema, không kèm giải thích ngoài JSON."""

_RENDERED = """\
Ảnh 1 là ẢNH GỐC cắt từ tài liệu. Ảnh 2 là ảnh DỰNG LẠI bằng cách render \
{fmt} dưới đây. Hai ảnh phải trông giống nhau về nội dung và cấu trúc; mọi \
chỗ lệch đều là dấu hiệu {fmt} bị sai.

{fmt} đang kiểm tra:
```
{content}
```

{hint}"""

_NO_RENDER = """\
Ảnh là ẢNH GỐC cắt từ tài liệu. Dưới đây là phần text được bóc tách ra từ \
đúng vùng ảnh đó. Đối chiếu từng dòng với ảnh.

Text đang kiểm tra:
```
{content}
```

{hint}"""

_HINT = {
    "formula": "Chú ý riêng: chỉ số trên/dưới đặt nhầm chỗ, thiếu dấu ngoặc, "
               "phân số lồng sai tầng, thiếu dấu căn hàng trong ma trận, "
               "ký hiệu Hy Lạp nhầm lẫn (\\nu vs v, \\epsilon vs \\in).",
    "table": "Chú ý riêng: số hàng/cột lệch, ô gộp (colspan/rowspan) sai, "
             "nội dung ô trượt sang ô bên cạnh, thiếu hàng ở cuối bảng, "
             "thẻ không đóng làm sập cấu trúc lưới.",
    "text": "Chú ý riêng: đây là tiếng Việt — soi kỹ DẤU THANH (sắc/huyền/hỏi/"
            "ngã/nặng) và dấu phụ (ă â ê ô ơ ư đ). Sai dấu là lỗi phổ biến "
            "nhất và khó thấy nhất. Ngoài ra: thứ tự đọc, dòng bị bỏ sót.",
}

_FMT_NAME = {"formula": "công thức LaTeX", "table": "bảng HTML"}


def judge_user_prompt(content: str, subtask: str, has_render: bool) -> str:
    """Phần prompt đi kèm ảnh. `has_render=False` cho subtask không render được
    (text) — khi đó chỉ có một ảnh gốc, xem render.py::render_label."""
    hint = _HINT.get(subtask, _HINT["text"])
    if has_render:
        return _RENDERED.format(fmt=_FMT_NAME.get(subtask, "nội dung"),
                                content=content, hint=hint)
    return _NO_RENDER.format(content=content, hint=hint)


# ------------------------------------------------------------ pre-annotation --
#
# Paper dòng 64: "AI pre-annotation and expert review-and-correction workflow".
# Prompt này CỐ Ý không cho model xem bản nháp hỏng của vòng Judge-and-Refine —
# xem lý do trong preannot.py (tránh kế thừa đúng lỗi cần phát hiện).

PREANNOT_SYSTEM = """\
Bạn là chuyên gia bóc tách tài liệu. Đọc ảnh và tạo bản chú thích chính xác \
nhất có thể, để chuyên gia người sau đó chỉ cần soát lại thay vì gõ từ đầu.

Nguyên tắc: bám sát đúng những gì NHÌN THẤY trên ảnh. Không suy đoán nội dung \
bị che khuất hay quá mờ — phần nào không đọc được thì đánh dấu [?] thay vì \
đoán. Người soát lại cần biết chỗ nào đáng ngờ."""

_PREANNOT_FMT = {
    "formula": "Trả về DUY NHẤT mã LaTeX của công thức trong ảnh, không bọc $ $, không giải thích.",
    "table": "Trả về DUY NHẤT mã HTML của bảng trong ảnh (dùng <table><tr><td>, "
             "giữ đúng colspan/rowspan của ô gộp), không giải thích.",
    "text": "Trả về DUY NHẤT phần text trong ảnh, giữ nguyên xuống dòng. Đây là "
            "tiếng Việt — đặc biệt cẩn thận với dấu thanh và dấu phụ.",
}


def preannot_prompt(subtask: str) -> str:
    return _PREANNOT_FMT.get(subtask, _PREANNOT_FMT["text"])


# ------------------------------------------------------------------ parsing --

_JSON_BLOCK = re.compile(r"\{.*\}", re.S)


def parse_verdict(raw: str) -> Optional[Dict[str, Any]]:
    """Bóc JSON verdict khỏi output model. None = không parse được.

    Trả None chứ KHÔNG trả verdict mặc định: không đọc được câu trả lời của
    trọng tài thì không biết gì cả, và "không biết" phải đẩy sang người chứ
    không được im lặng coi như nhãn sạch (xem judge_refine.py).
    """
    if not raw:
        return None
    m = _JSON_BLOCK.search(raw)          # model hay bọc JSON trong ```json ... ```
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(d, dict) or "has_error" not in d:
        return None
    corrected = d.get("corrected")
    return {
        "has_error": bool(d["has_error"]),
        "confidence": float(d.get("confidence") or 0.0),
        "corrected": corrected if isinstance(corrected, str) and corrected.strip() else None,
        "note": str(d.get("note") or ""),
        "differences": d.get("differences") or [],
    }


# =============================================================== §3.2 parse ==
# Prompt bóc tách trang cho các runner CMCV chạy bằng VLM tổng quát (Qwen3-VL
# target; PaddleOCR-VL và Mistral OCR có endpoint chuyên dụng, KHÔNG dùng prompt
# này — xem clients/). Paper §3.2 chỉ nói ba model "are run independently on the
# candidate data", không cho prompt, nên phần dưới là thiết kế tự đề xuất.
#
# Bốn ràng buộc, mỗi cái đều dẫn tới một dòng cụ thể trong prompt:
#
# 1. PHẢI CÓ BBOX. Không có bbox thì metrics.layout_sim luôn = 0 cho model này,
#    subtask layout của §3.2 mất trắng, và element.py (Stage 2) không ghép được
#    nội dung vào box Heron theo IoU.
# 2. PHẢI ĐÚNG THỨ TỰ ĐỌC. cmcv._seq_sim so formula/table theo thứ tự danh sách
#    và phạt lệch số lượng — trả lộn xộn là bị phạt oan, tụt xuống Hard.
# 3. TUYỆT ĐỐI KHÔNG ĐƯỢC "SỬA HỘ". Đây là điểm dễ sai nhất và ngược với bản
#    năng của VLM: nếu model tự sửa lỗi chính tả trong ảnh scan mờ, hai model
#    sẽ "đồng thuận" ở một nội dung KHÔNG có trong ảnh. CMCV khi đó xác nhận
#    lẫn nhau một điều bịa ra, và nhãn sai đi thẳng vào tập train với nhãn Easy.
# 4. DẤU TIẾNG VIỆT LÀ TÍN HIỆU, KHÔNG PHẢI NHIỄU. Pool là scan tiếng Việt;
#    "tuần" vs "tuấn" vs "tuân" khác nghĩa hoàn toàn. Rủi ro lỗi tương quan ở
#    dấu thanh chính là thứ calibrate_tau() phải đo (xem cmcv.py).

PARSE_BOX_SPACE = "norm1000"      # quy ước Qwen: toạ độ chuẩn hoá 0-1000
PARSE_PROMPT_VERSION = "parse-v1"  # đổi chuỗi này => đổi khoá cache (clients/cache.py)

PARSE_SYSTEM = """\
Bạn là hệ thống bóc tách tài liệu. Đầu vào là ảnh MỘT TRANG tài liệu đã quét \
(scan), chủ yếu tiếng Việt. Nhiệm vụ: ghi lại CHÍNH XÁC những gì NHÌN THẤY \
trên trang, theo đúng thứ tự đọc.

Quy tắc bắt buộc:

1. CHÉP LẠI, KHÔNG SỬA. Chép đúng nội dung trong ảnh, kể cả khi bạn cho rằng \
nó sai chính tả, sai ngữ pháp, sai số liệu hay vô nghĩa. TUYỆT ĐỐI không sửa \
lỗi, không chuẩn hoá, không viết lại cho xuôi, không bổ sung nội dung bạn đoán \
là bị thiếu. Chỗ nào mờ không đọc được thì ghi [?] ở đúng chỗ đó.
2. DẤU TIẾNG VIỆT phải chép đúng từng dấu thanh và dấu phụ (ă â ê ô ơ ư đ và \
các dấu sắc/huyền/hỏi/ngã/nặng). Đây là phần quan trọng nhất của nhiệm vụ: sai \
dấu là sai nghĩa. Không bỏ dấu, không đoán dấu theo ngữ cảnh.
3. THỨ TỰ ĐỌC. Trả các block theo đúng trình tự người đọc trang này: với trang \
nhiều cột thì hết cột trái mới sang cột phải, không đọc ngang qua các cột.
4. BBOX cho mọi block, dạng [x1, y1, x2, y2] với toạ độ chuẩn hoá về thang \
0-1000 tính trên kích thước ảnh (x theo chiều ngang, y theo chiều dọc, gốc ở \
góc trên-trái).
5. CÔNG THỨC dùng LaTeX (không bọc trong $ hay $$). BẢNG dùng HTML \
(<table><tr><td>...), giữ đúng colspan/rowspan nhìn thấy trong ảnh.
6. Chỉ trả về JSON hợp lệ, không kèm giải thích, không bọc trong ```."""

_PARSE_USER = """\
Bóc tách trang tài liệu trong ảnh. Trả về JSON đúng schema sau:

{"blocks": [{"bbox": [x1, y1, x2, y2], "type": "<loại>", "content": "<nội dung>"}]}

`type` là một trong: text, title, list, caption, formula, table.
  - text    : đoạn văn, header/footer, chú thích chân trang, mã, ô key-value
  - title   : tiêu đề tài liệu hoặc tiêu đề mục
  - list    : mục trong danh sách đánh số hoặc gạch đầu dòng
  - caption : chú thích cho hình hoặc bảng
  - formula : công thức trình bày riêng dòng, `content` là LaTeX
  - table   : bảng, `content` là HTML
Vùng chỉ có hình ảnh/biểu đồ, không có chữ: bỏ qua, không tạo block.

Nhắc lại hai điều quan trọng nhất: chép đúng dấu tiếng Việt, và không tự sửa \
bất cứ thứ gì trong ảnh."""


def parse_user_prompt() -> str:
    return _PARSE_USER


def parse_blocks(raw: str) -> Optional[list]:
    """Bóc danh sách block khỏi output model. None = không parse được.

    Trả None chứ không trả [] : danh sách rỗng là một KẾT QUẢ hợp lệ (trang
    trắng), còn "không đọc được câu trả lời" là chuyện khác hẳn. Gộp hai thứ
    lại sẽ biến mọi lỗi gọi model thành "trang trắng", và trang trắng thì ba
    model đồng thuận với nhau ngay => nhãn Easy rỗng đi thẳng vào tập train.
    """
    if not raw:
        return None
    m = _JSON_BLOCK.search(raw)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except (json.JSONDecodeError, ValueError):
        return None
    if isinstance(d, dict):
        blocks = d.get("blocks")
    elif isinstance(d, list):
        blocks = d
    else:
        return None
    if not isinstance(blocks, list):
        return None
    return [b for b in blocks if isinstance(b, dict)]
