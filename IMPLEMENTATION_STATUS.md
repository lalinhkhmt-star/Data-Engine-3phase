# Trạng thái implementation — §3.1 DDAS, §3.2 CMCV, §3.3 Judge-and-Refine

Đối chiếu với `minerU2.5pr.md`. Ghi lại ranh giới rõ ràng giữa "đã thiết kế/code
xong" và "đã kiểm chứng bằng dữ liệu/model thật" — hai việc khác nhau, đừng lẫn.

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
  Gemini 3 Pro API) — `cmcv.py`
- Taxonomy Easy/Medium/Hard neo theo target vs 2 external — `assign_tier()`
- Metrics NED/TEDS-proxy/CDM-proxy/layout-F1 — `metrics.py`

### Chưa kiểm chứng — quan trọng nhất, chặn mọi kết quả downstream

1. **`calibrate_tau()` chưa từng chạy trên dev-set có ground truth.**
   Đây là giả định CỐT LÕI của toàn bộ CMCV: "2+ model độc lập đồng thuận ⇒
   kết quả đúng". Giả định này **hoàn toàn chưa được đo** cho đúng 3 model
   đang dùng (Qwen3-VL-8b / Mistral OCR 4 / Gemini 3 Pro) trên tiếng Việt.
   Rủi ro cụ thể đã nêu trước đó: Mistral và Gemini có thể cùng mắc lỗi
   tương quan ở dấu thanh/dấu phụ tiếng Việt mà không model nào phát hiện ra
   model kia sai — nếu vậy, τ mặc định trong `config.py::CMCVConfig.tau`
   (0.85–0.95) sẽ cho kết quả sai mà không có cách nào biết được nếu không đo.
   → **Việc cần làm**: xây dev-set tiếng Việt nhỏ có ground truth (con người
   gán nhãn), chạy `calibrate_tau()`, nếu `P(đúng | đồng thuận) < 0.98` thì
   bật `require_3way_for_easy`.

2. **Chưa nối API/model thật** — `parse_fn`, `image_fn` trong `pipeline.py`
   vẫn là placeholder, mọi test trong dự án đến nay đều dùng `ParseResult`
   giả lập, chưa gọi Mistral OCR / Gemini 3 Pro / Qwen3-VL-8b thật lần nào.

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
| Chạy thử end-to-end | `testkit/demo_judge_refine.py` | `python3 -m ddas.testkit.demo_judge_refine` |

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
   có Qwen3-VL-30B, cùng dòng model. Ở repo này còn sai rõ hơn (target chính là
   Qwen3-VL-8B, Gemini 3 Pro đã làm trọng tài CMCV), nên mặc định đặt
   `JUDGE_MODEL = "gpt-5"` cho khác dòng hẳn.
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
