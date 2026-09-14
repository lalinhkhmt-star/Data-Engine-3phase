# Data Engine — Phần 1: DDAS

Triển khai Phần 1 (Section 3.1, *Diversity-and-Difficulty-Aware Sampling*) của Data Engine
kiểu MinerU2.5-Pro, kèm CMCV (3.2) ở mức đủ để Phần 1 chạy được.

```
ddas/
  config.py     cấu hình + ngân sách
  metrics.py    NED / TEDS / CDM-proxy / layout-F1  (CPU, O(n))
  cmcv.py       CMCV có cascade — bỏ qua model 30B khi 2 model rẻ đã đồng thuận
  cluster.py    K-Means phân cấp + kênh tail + khử trùng lặp LSH
  density.py    độ hiếm cục bộ + chẩn đoán tự bật/tắt
  probe.py      probe-and-extrapolate: trọng số cụm từ ~3% pool
  element.py    CMCV mức element dẫn xuất từ CMCV trang (0 GPU-hour thêm)
  sampler.py    phân bổ lồng nhau (cây cụm x độ khó) + water-filling
  embed_real.py    encoder ViT-base THẬT (HF transformers) + layout-prior + ghép đặc trưng
  layout_prior.py  đặc trưng hình học 24-d từ text layer PDF (PyMuPDF), không cần GPU
  testkit/eval_embedding.py   đo purity/NMI: ViT thuần vs layout thuần vs ghép — CHẠY TRÊN DỮ LIỆU THẬT
  pipeline.py   orchestrator 9 bước
  costmodel.py  mô hình chi phí GPU/lưu trữ
run_demo.py     benchmark trên corpus tổng hợp, 2 chế độ giả định
```

## Chạy thử

```bash
python3 run_demo.py                       # benchmark 4 chiến lược x 2 chế độ
python3 -c "from ddas.costmodel import *; print(render(ddas_plan(), 256))"
```

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

## Bốn thay đổi so với mô tả trong paper

| # | Thay đổi | Lý do | Đo được |
|---|----------|-------|---------|
| 1 | **Cascade CMCV** — chỉ gọi Qwen3-VL-30B khi MinerU và PaddleOCR bất đồng | Easy chỉ cần MinerU đồng thuận với *ít nhất một* external, nên khi 2 model rẻ đã khớp thì kết luận không đổi | −46.5% GPU-hours, nhãn **giống hệt** |
| 2 | **Probe-and-extrapolate** | Quyết định *lấy mẫu ở đâu* không cần CMCV toàn pool, chỉ cần phân bố độ khó mỗi cụm | CMCV cho khâu lấy mẫu chỉ trên ~3% pool, sai số ước lượng trung vị 0.04 |
| 3 | **Element-CMCV dẫn xuất** | Đầu ra trang đã chứa element; chỉ cần căn bbox (Hungarian) rồi áp lại độ đo | ~0 GPU-hour thay vì nhân 3 lượt suy luận trên ~1.8B element |
| 4 | **Phân bổ lồng nhau + rarity có chẩn đoán** | Một lần water-fill với `w = gain × w_cụm` khiến chiều độ khó (tỉ lệ 10:1) áp đảo chiều đa dạng (2-3:1) | xem bảng dưới |

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
   được đo trên dev-set có ground truth (`cmcv.calibrate_tau`), vì MinerU2.5 và PaddleOCR-VL
   có thể mắc **lỗi tương quan**. Nếu `P(đúng | đồng thuận) < 0.98`, bật
   `require_3way_for_easy` — cascade khi đó mất tác dụng và chi phí quay về mức đầy đủ.
# Data-Engine-3phase
