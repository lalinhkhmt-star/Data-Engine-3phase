# Kiểm chứng model nhúng cho DDAS — Dữ liệu & Hướng dẫn chạy

Ghi lại trạng thái đã chuẩn bị (dữ liệu + model đã tải) và cách chạy đánh giá
purity/NMI để chọn model nhúng trang cho `ddas/config.py::EmbedConfig`.

## 1. Vì sao cần kiểm chứng

`config.py` gợi ý `facebook/dinov2-base` cho tầng nhúng trang, nhưng đó là suy
đoán chưa kiểm chứng: DINOv2 pretrain trên ảnh tự nhiên (LVD-142M), không có
gì đảm bảo nó tách được cấu trúc bố cục tài liệu (số cột, bảng lồng nhau, mật
độ công thức) thay vì chỉ tách theo texture/mật độ pixel. Công cụ
`ddas/testkit/eval_embedding.py` đo trực tiếp bằng purity/NMI/silhouette trên
dữ liệu thật thay vì tin cấu hình mặc định.

## 2. Dữ liệu đã tải

Nguồn: [`docling-project/DocLayNet-v1.2`](https://huggingface.co/datasets/docling-project/DocLayNet-v1.2)
(HuggingFace, license CDLA-Permissive-1.0) — có PDF thật kèm text layer
(`pdf_cells`), và nhãn `metadata.doc_category` sẵn có cho 6 loại tài liệu.

| Thuộc tính | Giá trị |
|---|---|
| Vị trí | `/home/linhlt109/Documents/Data_Engine/data/doclaynet_sample/` |
| Tổng số | 300 PDF (50 trang / category × 6 category) |
| Dung lượng | 20 MB |
| Script tải | `ddas/testkit/fetch_doclaynet_sample.py` |

```
data/doclaynet_sample/
  financial_reports/       50 pdf
  government_tenders/      50 pdf
  laws_and_regulations/    50 pdf
  manuals/                 50 pdf
  patents/                 50 pdf
  scientific_articles/     50 pdf
```

**Lưu ý khi mở rộng mẫu:** stream của dataset này KHÔNG xáo trộn sẵn — 200
mẫu đầu tiên đọc tuần tự đều là `financial_reports`. Script đã bật
`.shuffle(buffer_size=...)` và hỗ trợ tiếp nối từ dữ liệu có sẵn (không tải
lại phần đã có):

```bash
python3 -m ddas.testkit.fetch_doclaynet_sample \
  --out data/doclaynet_sample --per-category 200 \
  --shuffle-buffer 10000 --max-scan 500000
```

## 3. Model nhúng đã tải sẵn (cache HuggingFace)

Vị trí cache: `~/.cache/huggingface/hub/` — copy nguyên thư mục này khi đẩy
lên server GPU để khỏi tải lại (hoặc set biến môi trường `HF_HOME` trỏ vào
đó); nếu server có mạng riêng thì không cần, `transformers` tự tải khi chạy
lần đầu.

| Model | Domain pretrain | Dung lượng | Vai trò trong so sánh |
|---|---|---|---|
| `facebook/dinov2-base` | Ảnh tự nhiên (LVD-142M) | 331M | Mặc định hiện tại trong `EmbedConfig` — cần kiểm chứng |
| `facebook/dinov2-small` | Ảnh tự nhiên | 85M | Bản nhẹ, để lặp thử nghiệm nhanh trên CPU |
| `openai/clip-vit-base-patch32` | Ảnh-text tự nhiên (caption) | 1.2G | Baseline khác hướng tự nhiên |
| `microsoft/dit-base` | **11M ảnh tài liệu scan thật (IIT-CDIP)** | 703M | **Đúng domain — phép so sánh có ý nghĩa nhất với DINOv2** |

Tổng dung lượng 4 model: **2.3 GB**.

`microsoft/dit-base` load qua kiến trúc BEiT, output `(batch, 197, 768)` —
196 patch token + 1 CLS token ở vị trí 0, giống hệt cấu trúc DINOv2 nên cắm
thẳng vào `ViTPageEncoder` (`ddas/embed_real.py`) không cần sửa code, chỉ đổi
tham số `--vit-model`.

**Chưa cắm được (cần sửa code thêm):** `naver-clova-ix/donut-base`,
`facebook/nougat-base` — cả hai dùng backbone Swin Transformer, không có CLS
token đơn giản ở vị trí 0 như ViT/BEiT, phải mean-pool patch token thủ công.

## 4. Cách chạy đánh giá

### Trên máy này (CPU) — kiểm tra code chạy đúng trước khi lên GPU

```bash
cd /home/linhlt109/Documents/Data_Engine

python3 -m ddas.testkit.eval_embedding \
  --data-dir data/doclaynet_sample \
  --vit-model facebook/dinov2-small \
  --dpi 120 \
  --out data/results_dinov2_small.json
```

### Trên server GPU — chạy đủ 4 model để so sánh

```bash
pip install torch transformers PyMuPDF scikit-learn pillow

for MODEL in facebook/dinov2-base facebook/dinov2-small \
             openai/clip-vit-base-patch32 microsoft/dit-base; do
  TAG=$(echo "$MODEL" | tr '/' '_')
  python3 -m ddas.testkit.eval_embedding \
    --data-dir data/doclaynet_sample \
    --vit-model "$MODEL" \
    --device cuda \
    --dpi 150 \
    --out "data/results_${TAG}.json"
done
```

Muốn test đúng chiều 512-d mà paper MinerU2.5-Pro mô tả: thêm `--pca-dim
512`. Muốn ép đúng số cụm bằng số category thật: thêm `--k 6`.

## 5. Đọc kết quả

Mỗi lần chạy in bảng so sánh 3 candidate cho model đó:

| candidate | Ý nghĩa |
|---|---|
| `vit` | Chỉ đặc trưng thị giác từ model đang test |
| `layout` | Chỉ đặc trưng hình học 24-d từ text layer PDF (`ddas/layout_prior.py`) — không cần GPU |
| `vit+layout` | Ghép cả hai, mỗi nguồn L2-normalize riêng trước khi ghép |

Chỉ số:
- **purity** — tỉ lệ điểm trong mỗi cụm K-Means khớp với nhãn `doc_category` chiếm đa số của cụm đó.
- **NMI** (normalized mutual information) — chỉ số chính để so sánh giữa các model, không lệ thuộc số cụm.
- **silhouette** — độ tách biệt hình học của các cụm (không cần nhãn thật).

**Cách kết luận:** so `NMI` của `vit` giữa 4 model. Nếu `dit-base` thắng rõ
`dinov2-base` → xác nhận giả thuyết ViT tự nhiên không đủ, nên đổi
`EmbedConfig.vit_name` sang model domain tài liệu. Nếu không thắng → giả
thuyết sai trên dữ liệu này, giữ nguyên DINOv2 hoặc dùng CLIP.

Đồng thời so `vit` vs `vit+layout` trong cùng một model: nếu `vit+layout`
luôn nhỉnh hơn thì `layout_prior.py` đáng giữ trong pipeline sản xuất dù
chọn ViT nào.

## 6. Giới hạn của bộ đo này

- 300 trang (50/category) là mẫu nhỏ — đủ xác nhận xu hướng ban đầu, không
  đủ để kết luận chắc chắn ở quy mô sản xuất. Nên chạy lại với
  `--per-category 200+` trên GPU trước khi quyết định cuối.
- `--pca-dim` (nếu dùng) fit PCA trên chính batch đánh giá — chỉ hợp lệ để
  so sánh nội bộ giữa các candidate trong CÙNG một lần chạy, không dùng để
  so sánh giữa các lần chạy khác nhau hoặc triển khai vào sản xuất.
- DocLayNet chỉ có 6 category tài liệu văn bản-nặng (financial, scientific,
  laws, tenders, manuals, patents) — không có category thiên về bảng dày đặc
  hay công thức dày đặc mà paper MinerU2.5-Pro nhấn mạnh là long-tail khó.
  Cân nhắc bổ sung PubTables-1M (bảng) hoặc PDF arXiv toán học (công thức)
  cho vòng đánh giá tiếp theo.
