# Trạng thái implementation — §3.1 DDAS, §3.2 CMCV, §3.3 Judge-and-Refine

Đối chiếu với `minerU2.5pr.md`. Ghi lại ranh giới rõ ràng giữa "đã thiết kế/code
xong" và "đã kiểm chứng bằng dữ liệu/model thật" — hai việc khác nhau, đừng lẫn.

## Mốc M0 — lớp nối model thật (`ddas/clients/`)

Trước mốc này, toàn bộ repo chạy bằng `ParseResult` giả lập: thuật toán đúng
nhưng **chưa gọi model nào lần nào**. `clients/` lấp bốn ranh giới core đã chừa
sẵn (`runners` của CMCV, `CallModel` của judge/pre-annotation, `image_fn`), và
không sửa một dòng nào trong core.

### Đã code + kiểm chứng qua HTTP thật (stub localhost, 22/22 kiểm tra)

`python3 -m ddas.testkit.demo_clients` dựng stub nói đúng giao thức của 4 nhà
cung cấp rồi chạy CMCV + vòng judge qua đó. Phủ: đường sạch, cascade cắt đúng
model thứ 3, tier MEDIUM, cache 0 lời gọi ở lần chạy lại, retry qua 503, output
hỏng, trang rỗng, verdict trọng tài.

### Hai lỗi thật phát hiện được nhờ bộ test này

1. **Cả 3 model cùng trả rỗng ⇒ tier EASY với nhãn RỖNG.** `cmcv.pair_sims` coi
   hai kết quả rỗng là giống nhau tuyệt đối (mọi sim = 1.0), nên trang rỗng
   được gán EASY và nhãn rỗng đi thẳng vào tập train dưới mác "model đồng
   thuận" — mà cascade còn cắt luôn model thứ 3 nên không có ai phản biện.
   Lỗi này **không crash, không có dấu hiệu nào trong log**. Ở quy mô triệu
   trang, một endpoint chập chờn 20 phút là đủ nhiễm bẩn hàng chục nghìn mẫu.
   → Đã chặn ở cả 3 runner (`allow_empty=False`, xem `normalize.is_empty_result`).
   Trang trắng không mang tín hiệu cho bất kỳ subtask nào trong 4 subtask nên
   loại nó không mất gì. Cổng đúng vẫn là `scanqa.py` chạy TRƯỚC mọi lời gọi
   model; đây là lớp chặn thứ hai.
2. **`build_clients` không cho ghi đè base URL của Mistral/OpenAI/Gemini**, nên
   test trỏ vào stub localhost vẫn lặng lẽ bắn ảnh ra API thật của nhà cung cấp
   với key giả. Chỉ nhận 401 nên không đổ vỡ và không có dấu hiệu gì.
   → Đã cho ghi đè qua tham số hoặc `MISTRAL_OCR_URL`/`OPENAI_BASE_URL`/
   `GEMINI_BASE_URL`, và `demo_clients` nay chặn cứng mọi kết nối ra ngoài
   localhost (`block_external_network()`).

### Chưa kiểm chứng — cần endpoint/key thật

1. **Shape response của Mistral OCR và PaddleOCR-VL chưa nhìn tận mắt lần nào.**
   Adapter viết phòng thủ (Mistral dò cả structured lẫn markdown; Paddle cho
   cắm `extract_fn`) nhưng mặc định là **suy đoán**. Sai shape ⇒ Paddle cách ly
   hàng loạt, hoặc Mistral mất bbox ⇒ `layout_sim` với nó luôn = 0 ⇒ subtask
   layout của §3.2 mất external rẻ.
   → **Việc cần làm**: `python3 -m ddas.testkit.preflight --pages <thư mục> --no-cache`,
   đọc bằng mắt bbox + text in ra, bật `strict_layout=True` nếu muốn Mistral
   thiếu bbox là lỗi cứng.
2. **Chưa đo throughput/chi phí thật** để đối chiếu `costmodel.py` (mốc M2 yêu
   cầu khớp ±30%). `EngineClients.stats()` đã đếm sẵn calls/token/latency.
3. **Batch API của Mistral (-50%) chưa hiện thực.** Luồng batch là bất đồng bộ,
   không khớp chữ ký `page_id -> ParseResult` của CMCV. Muốn dùng thì chạy một
   pass gom trước đổ vào cache, rồi CMCV chạy hoàn toàn bằng cache-hit. Ở quy
   mô 60M trang đây là khoản $120K.

## Đối chiếu lại với paper — 4 lỗi thật đã sửa

Rà soát từng khẳng định của `minerU2.5pr.md` so với code. Phần lớn khớp (xem
bảng "Sáu thay đổi" trong README cho các chỗ lệch CÓ CHỦ Ý). Bốn chỗ dưới đây
là SAI thật, không phải lệch có chủ ý — tất cả cùng một loại: **nội dung rỗng
hoặc thiếu bằng chứng bị tính thành "đồng thuận" rồi gán tier EASY.**

Loại lỗi này không crash, không có dấu hiệu trong log, và sinh ra nhãn huấn
luyện sai. Test chặn hồi quy: `python3 -m ddas.testkit.demo_elements` (18 kiểm
tra) và `demo_clients.py` bài [6b].

| # | Lỗi | Hệ quả | Sửa |
|---|---|---|---|
| 1 | `_elements_of` gán `content=""` cho MỌI element text, vì `ParseResult` chỉ giữ text dạng chuỗi ĐàGHÉP | Subtask `text` (**25M/60M = 42% dataset**) sinh 100% bản ghi SFT nhãn RỖNG tier EASY | Thêm `ParseResult.contents` (nội dung theo từng block, căn với `boxes`/`labels`); adapter trong `clients/normalize.py` điền sẵn |
| 2 | Phép ghép so nhãn đã-gộp (`"text"`) với nhãn anchor chưa gộp (`"title"`) nên không bao giờ khớp | Element `title`/`list`/`caption` luôn Hard, rồi rơi khỏi cả 3 subtask — **bị loại âm thầm khỏi Stage 2** | Giữ nhãn nguyên vẹn khi ghép, gộp về subtask SAU qua `SUBTASK_OF_LABEL` |
| 3 | Dòng vá `s_mp = 1.0 if not (ca or cb)` — hai nội dung rỗng thành đồng thuận tuyệt đối | Che đúng lỗi #1 nên nó không lộ ra | Gỡ dòng vá; cả hai cùng rỗng ⇒ thiếu bằng chứng ⇒ HARD |
| 4 | `assign_tier`: `s_mq is None` mặc nhiên trả EASY, giả định luôn là "cascade đã cắt" | Ở mức trang không lộ. Ở mức element, `cq=None` còn nghĩa model thứ 3 **không phủ bbox đó** ⇒ element bất đồng nặng vẫn nhận EASY kèm nhãn sai của target | Neo lại đúng điều kiện cascade dùng để cắt: chỉ EASY khi `s_mp >= tau` |

Lỗi #4 nguy hiểm nhất vì nó nằm ở hàm dùng chung cho cả mức trang lẫn mức
element, và chỉ sai ở một trong hai đường.

## §3.1 Diversity-and-Difficulty-Aware Sampling

### Đã code + test (dữ liệu giả, chạy sạch end-to-end)

| Bước paper | File | Ghi chú |
|---|---|---|
| Stage 1 — page-level | `cluster.py`, `probe.py`, `pipeline.py::run()` | K-Means phân cấp + probe CMCV + trọng số cụm theo độ khó |
| Stage 2 — element-level | `layout_heron.py`, `element.py` | Layout detection (Heron) chạy trước, độc lập với 3 model CMCV — đúng thứ tự Figure 3, không phải suy ra bbox bằng cách match 2 model CMCV |
| Final sampling — gộp 4 subtask | `sampler.py`, `sft.py` | `assemble_sft_set()` tra pseudo-label theo tier, gộp layout+text+formula+table; Hard/Invalid tách hàng đợi riêng, không lẫn vào tập train |

### Chưa kiểm chứng — cần làm trước khi tin kết quả

1. **`EmbedConfig.vit_name` (config.py:11) đang SAI theo dữ liệu đã đo thật.**
   `data/EMBEDDING_EVAL_REPORT.md` (mục 7, chạy GPU thật trên 300 trang DocLayNet,
   commit `1625bd5`) cho thấy `openai/clip-vit-base-patch32` (NMI 0.403) thắng
   gần gấp đôi `facebook/dinov2-base` (NMI 0.196) — nhưng config vẫn chưa đổi
   theo kết quả này.
   → **Việc cần làm**: đổi `vit_name` sang CLIP, hoặc ghi rõ lý do nếu cố tình giữ DINOv2.

2. **Chưa có eval embedding nào chạy trên dữ liệu tiếng Việt.**
   `data/vietnamese_sample/` (invoices/receipts/theses) đã tải ảnh nhưng
   **chưa có file `results_*.json` nào cho tập này** — kết luận CLIP thắng ở
   trên chỉ đo trên DocLayNet (tiếng Anh, 6 category tài liệu chung), không
   phải layout tiếng Việt.
   → **Việc cần làm**: chạy `ddas.testkit.eval_embedding --data-dir
   data/vietnamese_sample` với cả 4 model, so kết quả với bảng ở mục 7 của
   `EMBEDDING_EVAL_REPORT.md`.

3. **`HeronLayoutDetector.detect()` chưa chạy model thật lần nào.**
   Logic thuần Python (`drop_redundant_figure_wrappers`,
   `layout_prior_from_heron`) đã unit-test kỹ bằng `LayoutBox` giả lập, nhưng
   phần gọi `RTDetrV2ForObjectDetection` qua `transformers` (trong
   `_lazy_load()`/`detect()`) chưa từng tải model hay chạy trên ảnh thật.
   → **Việc cần làm**: chạy thử `HeronLayoutDetector().detect([ảnh thật])`
   trên vài trang scan tiếng Việt, kiểm tra bbox/class có hợp lý không.

## §3.2 Cross-Model Consistency Verification

### Đã code + test (dữ liệu giả)

- Cascade 3-model (target Qwen3-VL-8b tự host, cheap Mistral OCR 4 API, expensive
  PaddleOCR-VL tự host) — `cmcv.py`. Lịch sử đổi model thứ 3: Gemini 3 Pro (bản
  đầu) → Claude Opus 5 (giải phóng Gemini cho pre-annotation ở §3.3, dòng 64
  paper) → **PaddleOCR-VL** (theo yêu cầu không dùng model đóng-quyền/trả phí
  cho vai này; cân nhắc Chandra OCR trước nhưng loại vì kiến trúc dựa trên
  Qwen3VL — cùng lineage với target, phá vỡ tính độc lập của trọng tài).
  PaddleOCR-VL tự host (0.9B tham số, ERNIE-4.5) nên KHÔNG còn dòng chi phí API
  nào cho model thứ 3 — xem costmodel.py.
- Taxonomy Easy/Medium/Hard neo theo target vs 2 external — `assign_tier()`
- Metrics NED/TEDS-proxy/CDM-proxy/layout-F1 — `metrics.py`

### Chưa kiểm chứng — quan trọng nhất, chặn mọi kết quả downstream

1. **`calibrate_tau()` chưa từng chạy trên dev-set có ground truth.**
   Đây là giả định CỐT LÕI của toàn bộ CMCV: "2+ model độc lập đồng thuận ⇒
   kết quả đúng". Giả định này **hoàn toàn chưa được đo** cho đúng 3 model
   đang dùng (Qwen3-VL-8b / Mistral OCR 4 / PaddleOCR-VL) trên tiếng Việt.
   Rủi ro cụ thể đã nêu trước đó: Mistral và PaddleOCR-VL có thể cùng mắc lỗi
   tương quan ở dấu thanh/dấu phụ tiếng Việt mà không model nào phát hiện ra
   model kia sai — nếu vậy, τ mặc định trong `config.py::CMCVConfig.tau`
   (0.85–0.95) sẽ cho kết quả sai mà không có cách nào biết được nếu không đo.
   → **Việc cần làm**: xây dev-set tiếng Việt nhỏ có ground truth (con người
   gán nhãn), chạy `calibrate_tau()`, nếu `P(đúng | đồng thuận) < 0.98` thì
   bật `require_3way_for_easy`.

2. **Chưa nối API/model thật** — `parse_fn`, `image_fn` trong `pipeline.py`
   vẫn là placeholder, mọi test trong dự án đến nay đều dùng `ParseResult`
   giả lập, chưa gọi Mistral OCR / PaddleOCR-VL / Qwen3-VL-8b thật lần nào.

3. **Throughput/giá trong `costmodel.py` là ước lượng**, đánh dấu rõ "CHƯA ĐO
   THẬT" trong code — cần benchmark lại trên cụm máy thật và kiểm tra giá API
   hiện hành trước khi dùng để quyết định ngân sách.

## §3.3 Annotation Pipeline for Hard Case

### Đã code + test (trọng tài giả lập, render chạy thật)

| Bước paper | File | Ghi chú |
|---|---|---|
| Render-then-verify (dòng 51-53) | `render.py` | LaTeX: `pdflatex` → `mathtext`; HTML table: `pymupdf.Story`. Bảng tràn nhiều trang được ghép dọc, không cắt cụt |
| Vòng Judge-and-Refine (dòng 55) | `judge_refine.py::JudgeRefine` | 6 lối thoát: sửa xong / render lỗi / không đề xuất được bản sửa / sửa không đổi / quay vòng / hết vòng |
| Ưu tiên chú thích tay (dòng 61-62) | `judge_refine.py::prioritize` | Tiêu chí #1 correction efficiency, #2 marginal impact qua `weakness_by_subtask()` — tái dùng `sims` của §3.2, không đo thêm |
| Nối vào orchestrator | `pipeline.py::run_judge_refine()` | Tiêu thụ `hard_queue` của `assemble_sft_set()`; `RefinedRecord.to_sft()` giữ tier HARD + `label_source="judge-refine:*"` để truy vết |
| Judge-and-refine prompt (dòng 53) | `prompts.py` | Paper KHÔNG cho nội dung prompt. Tự thiết kế theo 3 ràng buộc rút từ chính paper: bắt liệt kê khác biệt thị giác TRƯỚC khi kết luận (chống thiên lệch dòng 49), neo vào so-hai-ảnh chứ không đọc-lại-chuỗi (dòng 51), bắt buộc khoanh vùng lỗi trong `note` (phục vụ ưu tiên dòng 61). Output JSON không parse được ⇒ KHÔNG coi là sạch |
| AI pre-annotation (dòng 64) | `preannot.py` | Gemini 3 Pro đúng như paper (hợp lệ vì không nằm trong pool CMCV). Pre-annotate TỪ ẢNH GỐC, cố ý không cho xem bản nháp hỏng ⇒ ý kiến độc lập thật; người nhận 2 phương án + chỗ khoanh lỗi, phân luồng soát nhanh/phân xử/gõ lại |
| Automated QA tools (dòng 64) | `annot_qa.py` | Paper chỉ có 1 câu, không nói QA gì. Tự chọn 3 nhóm tất định, không dùng model: hợp lệ về dạng (render lại được không) / tỉnh táo nội dung (rỗng, y hệt bản đã biết sai, dài bất thường, còn `[?]`) / nhất quán giữa annotator (trộn lặp 5%) |
| Phân tầng Stage 2 vs Stage 3 (dòng 66) | `sft.py::split_training_stages()` | Paper nói 192K dùng cho cả SFT lẫn GRPO nhưng không cho tiêu chí chia. Tự suy: GRPO cần reward chấm được TỰ ĐỘNG ⇒ chỉ formula/table vào GRPO (compile/dựng lưới kiểm được), text/layout về SFT; trong nhóm đó ưu tiên subtask model yếu nhất |
| Chạy thử end-to-end | `testkit/demo_judge_refine.py` | `python3 -m ddas.testkit.demo_judge_refine` — phủ cả 4 mảng trên |

### Bổ sung ngoài paper — bắt buộc phải có mới chạy thật được

| Vấn đề | File | Vì sao |
|---|---|---|
| **Trần ngân sách §3.3** | `config.py::JudgeRefineConfig.judge_budget`, `judge_refine.py::select_for_judging()` | Paper nói §3.3 xử lý "Hard samples" mà không nói bao nhiêu. Chạy đúng chữ ở quy mô 60M trang = 216M mẫu Hard → **$24.1M** riêng tiền trọng tài, gấp 83 lần cả §3.1+§3.2. Đặt trần 400K (đủ nuôi 192K mẫu người) → **$275K**. Mẫu vượt trần nằm lại hàng đợi, không vứt |
| **Cổng chất lượng ảnh scan** | `scanqa.py` | Pool 100% scan làm "khó" tách hai loại ngược nhau: khó vì cấu trúc (quý) vs khó vì ảnh hỏng (rác). CMCV gộp cả hai vào Hard rồi §3.3 đem cả hai đi tiêu tiền. Đo mờ/phân giải/mực/tương phản/nghiêng, cắm qua `validity_fn` có sẵn |
| **Xuất dữ liệu ra đĩa** | `io.py` | `pipeline.py:3` viết "checkpoint ra parquet" nhưng không dòng code nào làm — chạy xong mất sạch. JSONL (+gzip), `export_dataset()` xuất Stage 1/2/3 kèm manifest |
| **Chi phí §3.3** | `costmodel.py::judge_refine_plan()` | `ddas_plan()` dừng ở stage 2, thiếu đúng phần đắt nhất |

### Chưa kiểm chứng — cần làm trước khi tin kết quả

1. **Chưa gọi model trọng tài thật lần nào.** Toàn bộ test dùng `FakeJudge`
   kịch bản cố định — nó chỉ chứng minh luồng dữ liệu và 6 lối thoát chạy đúng,
   **không** nói gì về việc render-then-verify có thật sự giúp model phát hiện
   lỗi cấu trúc hay không. Đó chính là giả định trung tâm của §3.3.
   → **Việc cần làm**: chạy trên vài trăm mẫu Hard thật có ground truth, đo
   `resolve_rate` và — quan trọng hơn — tỉ lệ "sửa xong nhưng vẫn SAI" (trọng
   tài nhận nhầm là sạch). Chỉ số thứ hai mới quyết định có được phép đưa
   `RefinedRecord` vào tập train hay không; hiện chưa có gì đảm bảo nó thấp.

2. **`JUDGE_MODEL` đang CỐ Ý khác paper.** Paper dùng Qwen3-VL-235B và nói nó
   "độc lập với CMCV model pool" — lập luận này không đứng vững: pool của paper
   có Qwen3-VL-30B, cùng dòng model. Ở repo này, pool CMCV là Qwen3-VL-8B
   (target) / Mistral OCR (cheap) / PaddleOCR-VL (expensive), nên mặc định đặt
   `JUDGE_MODEL = "gpt-5"` (OpenAI) — dòng thứ tư, khác cả ba. Gemini 3 Pro
   (dòng thứ năm) CHỦ Ý không dùng ở đây, để dành cho pre-annotation (mục dưới).
   → **Việc cần làm**: hoặc đo lỗi tương quan giữa trọng tài và pool CMCV rồi
   giữ nguyên, hoặc sửa lại đoạn văn §3.3 trong paper cho khớp lựa chọn thật.

3. **Backend `mathtext` không phải LaTeX đầy đủ.** Máy hiện tại chưa cài TeX,
   nên mọi `\begin{bmatrix}`, `\begin{array}`, `\begin{cases}`... đều render
   lỗi *dù công thức hợp lệ* — tức là một phần Hard sẽ bị đẩy sang người vì lý
   do công cụ, không phải vì nhãn sai. `ExpertItem.backend` có ghi lại backend
   để chạy lại được, nhưng số liệu `render_failed` đo lúc này KHÔNG dùng để báo
   cáo được.
   → **Việc cần làm**: cài TeX (`pdflatex`) trước khi chạy thật; `render.py` tự
   ưu tiên nó khi có trong PATH.

4. **`max_rounds=3` và `converge_tau=0.995` là phỏng đoán.** Paper không cho số
   vòng. Chưa có dữ liệu để biết vòng thứ 3 còn cứu thêm được bao nhiêu mẫu so
   với chi phí gọi model.
   → **Việc cần làm**: chạy với `max_rounds` lớn rồi vẽ đường tỉ lệ cứu được
   theo số vòng, cắt ở chỗ đường nằm ngang.

5. **`expert_budget = 192_000` lấy thẳng từ paper**, chưa phải con số suy ra từ
   quy mô pool thật của dự án này.
