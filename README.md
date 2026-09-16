# Data Engine — Phần 1: DDAS

Triển khai Phần 1 (Section 3.1, *Diversity-and-Difficulty-Aware Sampling*) của Data Engine
kiểu MinerU2.5-Pro, kèm CMCV (3.2) ở mức đủ để Phần 1 chạy được, và §3.3
(*Annotation Pipeline for Hard Case*) xử lý tiếp hàng đợi Hard mà 3.2 để lại.

```
ddas/
  config.py     cấu hình + ngân sách
  metrics.py    NED / TEDS / CDM-proxy / layout-F1  (CPU, O(n))
  cmcv.py       CMCV có cascade — bỏ qua model thứ 3 (PaddleOCR-VL, tự host) khi target
                (Qwen3-VL, tự host) và Mistral OCR (API) đã đồng thuận
  cluster.py    K-Means phân cấp + kênh tail + khử trùng lặp LSH
  density.py    độ hiếm cục bộ + chẩn đoán tự bật/tắt
  probe.py      probe-and-extrapolate: trọng số cụm từ ~3% pool
  element.py    Stage 2 đầy đủ: layout detection (Heron) CHẠY TRƯỚC ra bbox+class
                độc lập, rồi tra nội dung target/cheap/expensive theo IoU (0 suy luận
                CMCV thêm) + crop-embed + cluster riêng theo loại element + lấy mẫu
                lồng nhau cho text/formula/table (tách quota khỏi layout)
  sampler.py    phân bổ lồng nhau (cây cụm x độ khó) + water-filling
  embed_real.py    encoder ViT-base THẬT (HF transformers) + layout-prior + ghép đặc trưng
  layout_prior.py  đặc trưng hình học 24-d từ text layer PDF (PyMuPDF), không cần GPU
  layout_heron.py  Docling Layout Heron v0 (RT-DETRv2, GPU) — 2 vai trò: layout-prior
                24-d cho trang scan ở Stage 1, VÀ nguồn bbox+class cho Stage 2 (element.py)
  testkit/eval_embedding.py   đo purity/NMI: ViT thuần vs layout thuần vs ghép — CHẠY TRÊN DỮ LIỆU THẬT
  pipeline.py   orchestrator — run() cho layout (mức trang), run_elements() cho
                text/formula/table (mức element), assemble_sft_set() gộp cả 4 subtask
  sft.py        Final sampling: tra pseudo-label đúng theo tier (Easy→target,
                Medium→cheap-external), gộp layout+text+formula+table thành 1 dataset;
                Hard/Invalid tách sang hàng đợi riêng (chưa có nhãn tin cậy)
  render.py     §3.3 render-then-verify: LaTeX→ảnh (pdflatex/mathtext), HTML table→ảnh
                (pymupdf.Story) — biến lỗi cấu trúc thành khác biệt nhìn thấy được
  judge_refine.py  §3.3 vòng soi-và-sửa nhãn Hard bằng model trọng tài khác dòng với cả
                3 model CMCV; phần không tự cứu được thì xếp ưu tiên sang chú thích tay
  prompts.py    §3.3 prompt trọng tài + pre-annotation (paper không cho nội dung, tự
                thiết kế theo 3 ràng buộc rút từ chính lập luận dòng 49/51/61)
  preannot.py   §3.3 AI pre-annotation ĐỘC LẬP (không cho model xem bản nháp hỏng) +
                gói việc cho chuyên gia: 2 phương án + chỗ khoanh lỗi -> soát nhanh/phân xử/gõ lại
  annot_qa.py   §3.3 QA nhãn người: hợp lệ về dạng (render lại được không) + tỉnh táo
                nội dung + nhất quán giữa annotator (trộn lặp 5% mẫu)
  scanqa.py     cổng chất lượng ảnh SCAN, chạy trước mọi lời gọi model (CPU): mờ/
                phân giải/mực/tương phản/nghiêng — tách "khó vì cấu trúc" (quý) khỏi
                "khó vì ảnh hỏng" (rác). Cắm vào CMCV qua validity_fn
  io.py         ghi/đọc JSONL + export_dataset() xuất bộ phân tầng Stage 1/2/3 kèm
                manifest — thiếu cái này thì chạy xong là mất sạch
  costmodel.py  mô hình chi phí — GPU-giờ (target tự host) + USD/trang (2 external qua API)
  clients/      LỚP NỐI MODEL THẬT (mốc M0) — trước đây toàn bộ repo chạy bằng
                ParseResult giả lập, chưa gọi model nào lần nào
    base.py         retry phân biệt lỗi tạm thời/lỗi của ta + token-bucket
                    rate-limit + circuit-breaker + đếm chi phí thật
    cache.py        cache đĩa khoá theo NỘI DUNG (model+prompt+bytes ảnh) —
                    bắt buộc: chạy lại pipeline không được trả tiền API lần hai
    pagestore.py    page_id -> ảnh (ảnh rời hoặc "file.pdf#7"), LRU, chuẩn hoá
                    cỡ ảnh tất định để khoá cache ổn định
    normalize.py    quy output 3 nhà cung cấp về 1 ParseResult: bản đồ nhãn về
                    taxonomy chung (layout_sim đòi khớp chuỗi CHÍNH XÁC) + đổi
                    bbox về pixel tuyệt đối của ảnh PageStore trả về
    qwen_vl.py      TARGET_MODEL qua vLLM     (OpenAI chat-completions)
    mistral_ocr.py  CHEAP_EXTERNAL qua API    (dò cả shape structured lẫn markdown)
    paddle_vl.py    EXPENSIVE_EXTERNAL tự host (extract_fn cắm được theo cách deploy)
    openai_compat.py  dùng chung cho Qwen tự host và GPT-5 trọng tài §3.3
    gemini.py       Gemini 3 Pro — CHỈ cho pre-annotation §3.3
run_demo.py     benchmark trên corpus tổng hợp, 2 chế độ giả định
```

## Chạy thử

```bash
python3 run_demo.py                       # benchmark 4 chiến lược x 2 chế độ
python3 -m ddas.testkit.demo_judge_refine  # §3.3: vòng judge-and-refine, trọng tài giả lập
python3 -m ddas.testkit.demo_clients       # lớp client: 22 kiểm tra qua HTTP thật trên localhost
python3 -m ddas.testkit.demo_elements      # Stage 2 mức element: 18 kiểm tra (chặn "đồng thuận rỗng")
python3 -c "from ddas.costmodel import *; print(render(ddas_plan(), 256))"
```

## Web UI — test pipeline trên 1 ảnh, xem từng giai đoạn + log real-time

```bash
pip install fastapi uvicorn python-multipart   # đã có sẵn trong môi trường này
export QWEN_BASE_URL=... MISTRAL_API_KEY=... PADDLE_VL_URL=... OPENAI_API_KEY=... GEMINI_API_KEY=...
uvicorn ddas.webapp.app:app --reload --port 8000
# mở http://localhost:8000
```

Kéo thả 1 ảnh trang, bấm "Chạy pipeline" — trang hiển thị 9 cột (mỗi giai
đoạn 1 cột: ingest, scanqa, embed, layout, cmcv, elements, judge_refine,
preannot, tổng kết), log đổ vào real-time qua SSE khi từng bước chạy xong.
Badge trên đầu trang báo ngay vai nào đã cấu hình trước khi bấm chạy.

Đây là công cụ TEST/DEV — không auth, 1 process, không thiết kế cho nhiều
người dùng đồng thời. Vai nào thiếu key thì giai đoạn đó tự báo và bỏ qua
(không giả lập, không âm thầm coi như sạch) — xem `ddas/webapp/runner.py`.
Bước quy mô pool (cluster/probe/expand) không áp dụng cho 1 ảnh nên bỏ qua,
bắt đầu thẳng từ scanqa.

## Chạy trên model thật

`demo_clients.py` dựng stub HTTP nói đúng giao thức của 4 nhà cung cấp và chạy
cả CMCV lẫn vòng judge qua đó — kiểm được retry/cache/circuit-breaker/hệ toạ độ
mà không tốn tiền, và có chốt chặn mọi kết nối ra ngoài localhost.

Khi đã có endpoint/key thật, chạy preflight TRƯỚC mỗi đợt lớn:

```bash
export QWEN_BASE_URL=http://<vllm-host>:8000/v1   # target, tự host
export MISTRAL_API_KEY=...                        # cheap external
export PADDLE_VL_URL=http://<host>:8080/layout-parsing   # expensive external, tự host
export OPENAI_API_KEY=...                         # trọng tài §3.3
export GEMINI_API_KEY=...                         # pre-annotation §3.3

python3 -m ddas.testkit.preflight --pages data/vietnamese_sample/invoices --no-cache
```

Preflight kiểm ba loại lỗi chỉ lộ ra khi gọi thật và đều hỏng ÂM THẦM: shape
response khác dự đoán, hệ toạ độ bbox sai, và model tự "sửa hộ" nội dung. Nó
KHÔNG thay thế `calibrate_tau()` — preflight trả lời "đường ống có thông
không", không trả lời "đồng thuận có nghĩa là đúng không".

## Kiểm chứng model nhúng bằng dữ liệu thật

`config.py` gợi ý DINOv2-base cho tầng nhúng, nhưng **đó là suy đoán chưa kiểm chứng**:
DINOv2 huấn luyện trên ảnh tự nhiên, không có gì đảm bảo nó tách được cấu trúc bố cục
tài liệu (số cột, bảng lồng nhau, mật độ công thức) thay vì chỉ tách theo texture/mật độ
pixel. `testkit/eval_embedding.py` đo trực tiếp trên dữ liệu của bạn — đừng tin cấu hình
mặc định tới khi chạy được cái này.

```bash
pip install torch transformers PyMuPDF scikit-learn pillow    # đã có sẵn trong môi trường này

# Tổ chức dữ liệu: mỗi thư mục con là MỘT LOẠI TRANG bạn biết trước (ground truth)
# data/
#   paper_1cot/        *.pdf hoặc *.png/*.jpg
#   bao_cao_da_cot/     *.pdf
#   bang_long_nhau/     *.pdf
#   cong_thuc_day_dac/  *.pdf
# >= 4 loại, mỗi loại >= 20 trang, các loại phải thực sự khác nhau về cấu trúc.

python3 -m ddas.testkit.eval_embedding --data-dir data/ --dpi 150

# Không có nhãn sẵn (chỉ đo silhouette, không đo được purity/NMI):
python3 -m ddas.testkit.eval_embedding --data-dir data_khong_nhan/ --unsupervised --k 8

# So model khác, hoặc PCA trước khi so:
python3 -m ddas.testkit.eval_embedding --data-dir data/ --vit-model facebook/dinov2-small --pca-dim 128
```

Output là bảng purity/NMI/silhouette cho 3 candidate (`vit` thuần, `layout` thuần,
`vit+layout` ghép), kèm ma trận nhầm lẫn để soi cụm nào bị trộn. Đã smoke-test end-to-end
bằng 60 PDF thật (4 loại bố cục dựng qua PyMuPDF) trên `facebook/dinov2-small` — code
chạy đúng, không lỗi ở mọi nhánh (có nhãn / không nhãn / PCA / input ảnh thô / trang không
có text layer). **NMI=1.0 tuyệt đối trên bộ test đó không chứng minh DINOv2 đủ tốt** — 4
loại bố cục dựng tay khác biệt quá lộ liễu (bảng có viền vẽ, công thức dùng ký hiệu Unicode
riêng). Kết luận thật về việc có cần layout-prior hay không phải chạy trên dữ liệu PDF thật
của bạn, nơi khác biệt giữa các loại tinh tế hơn nhiều.

## Sáu thay đổi so với mô tả trong paper

| # | Thay đổi | Lý do | Đo được |
|---|----------|-------|---------|
| 1 | **Cascade CMCV** — chỉ gọi PaddleOCR-VL khi Qwen3-VL (target) và Mistral OCR bất đồng | Easy chỉ cần target đồng thuận với *ít nhất một* external, nên khi 2 model rẻ đã khớp thì kết luận không đổi | −46.5% GPU-giờ của model thứ 3, nhãn **giống hệt** |
| 2 | **Empirical-Bayes shrink cho trọng số cụm** | Bản thân việc dò một mẫu nhỏ mỗi cụm rồi mở rộng LÀ mô tả của paper ("*An initial uniform sample from each cluster is evaluated by page-level CMCV*"), không phải thay đổi — chỉ phần co ước lượng về prior toàn cục mới là bổ sung | sai số ước lượng trung vị 0.04 trên ~3% pool |
| 3 | **Element-CMCV dẫn xuất** | Layout detection (Heron) chạy 1 lần/trang candidate ra bbox+class; nội dung tra theo IoU từ output CMCV trang đã có sẵn, không suy luận 3 model CMCV thêm lần nào | 0 lượt suy luận CMCV thêm trên ~1.8B element (chỉ tốn 1 lượt Heron/trang, xem costmodel.py) thay vì nhân 3 |
| 4 | **Phân bổ lồng nhau + rarity có chẩn đoán** | Một lần water-fill với `w = gain × w_cụm` khiến chiều độ khó (tỉ lệ 10:1) áp đảo chiều đa dạng (2-3:1) | xem bảng dưới |
| 5 | **Layout detector = Docling Layout Heron**, không phải "MinerU2.5 and PaddleOCR-VL layout detection models" như paper | Cần một nguồn bbox ĐỘC LẬP với cả 3 model CMCV. Dùng chính PaddleOCR-VL (đang là model thứ 3 của pool CMCV) làm layout detector sẽ phá vỡ tính độc lập đó | 17 lớp, xem layout_heron.py |
| 6 | **TEDS/CDM dùng bản proxy CPU O(n)**, không phải apted-TEDS / CDM render-based như paper | Phải chạy trên hàng chục triệu trang; bản chính thức cắm qua cùng chữ ký hàm khi cần | HỆ QUẢ: τ hiệu chuẩn trên proxy chỉ nhất quán nội bộ, **không so sánh được với số của bất kỳ paper nào** |

## Kết quả benchmark (ngân sách 30K mẫu, pool 300K, Zipf(1.25))

Chế độ **R1** (lớp phổ biến đồng nhất, lớp hiếm dị biệt — giống pool tài liệu thật):

| chiến lược | đa dạng hữu hiệu | %kb hiếm | %kb đầu | %rác | mật độ tín hiệu |
|---|---|---|---|---|---|
| Uniform | 10.6 | 8.9 | 35.8 | 6.1 | 0.280 |
| Cluster-only | 10.2 (×0.96) | 15.9 (×1.79) | 37.7 | 0.0 | 0.534 (×1.90) |
| DDAS (cụm × khó) | 10.3 (×0.96) | 15.7 (×1.77) | 37.3 | 0.0 | **0.783 (×2.79)** |
| DDAS + rarity (auto) | **16.2 (×1.53)** | **31.8 (×3.58)** | **22.5 (×0.63)** | 0.0 | 0.740 (×2.64) |

Chế độ **R2** (độ trải rộng không tương quan với tần suất): chẩn đoán tự **tắt** rarity,
kết quả về đúng dòng "DDAS (cụm × khó)" — ×2.78 tín hiệu, không hồi quy.

## Hai giới hạn cần biết

1. **Chiều đa dạng mong manh hơn chiều độ khó.** Chiều độ khó ổn định ×2.8 ở cả hai chế độ.
   Chiều đa dạng dựa trên *số lượng cụm* gần như không có tác dụng (×0.90–0.96), vì K-Means
   đặt centroid tỉ lệ với mật độ: lớp tần suất cao chiếm ~43% số cụm và mọi cách "chia đều
   theo cụm" đều kế thừa nguyên độ lệch đó. Chỉ hệ số **độ thưa cục bộ** mới sửa được, và
   chỉ khi giả định của nó đúng trên pool — nên nó được **đo trước rồi mới bật**
   (`density.rarity_regime_check`), không bật mặc định.
2. **CMCV chưa được hiệu chuẩn thì chưa dùng được.** Giả định "đồng thuận ⇒ đúng" phải
   được đo trên dev-set có ground truth (`cmcv.calibrate_tau`), vì Qwen3-VL (target) và
   Mistral OCR có thể mắc **lỗi tương quan** — nhất là trên tiếng Việt, nơi cả hai chưa
   chắc đã được kiểm chứng độc lập. Nếu `P(đúng | đồng thuận) < 0.98`, bật
   `require_3way_for_easy` — cascade khi đó mất tác dụng và chi phí quay về mức đầy đủ.
# Data-Engine-3phase
