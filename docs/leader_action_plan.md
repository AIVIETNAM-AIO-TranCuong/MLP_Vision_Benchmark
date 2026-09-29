# 🎯 Kế Hoạch Hành Động Cho Cường (Leader + Phụ trách gMLP)

> **Ngày hiện tại**: Thứ Năm, 24/09/2026  
> **Trạng thái**: Tuần 1 sắp kết thúc — Outline ✅ Paper Tracker ✅

---

## 📍 Đánh giá hiện trạng

| Deliverable Tuần 1 | Trạng thái |
|---------------------|-----------|
| Outline and Plan | ✅ Đã hoàn thành |
| Paper Tracker (9 papers) | ✅ Đã điền chi tiết |
| Câu hỏi nghiên cứu | ✅ Đã chốt 3 câu hỏi |
| Thiết kế thí nghiệm | ✅ Đã có (5 model × 3 mức × 2 seed) |

> [!IMPORTANT]
> Bạn có **2 mũ** cần đội song song: **🎩 Leader** (quản lý nhóm) và **🧪 Contributor** (code gMLP). Dưới đây tách rõ việc của từng mũ.

---

## 📅 TUẦN 1 — Còn lại (24/09 – 28/09)

### Thứ Năm 24/09 — Chốt & Chuẩn bị chuyển sang Tuần 2

#### 🎩 Leader
- [ ] **Gửi tin nhắn tổng hợp cho nhóm**, bao gồm:
  - Confirm lại phân công (ai làm model nào)
  - Thống nhất **cấu hình chung**: patch size = 4×4, 3 mức params (~1M, ~3M, ~8M)
  - Thống nhất **training recipe**: AdamW, LR=1e-3, cosine + warmup, random crop + flip, batch 128/256
  - Yêu cầu mỗi người đọc xong paper phụ trách → confirm hiểu kiến trúc
- [ ] **Tạo shared repo** (Google Drive hoặc GitHub) với cấu trúc:
  ```
  MD4_conquer/
  ├── models/          # Code từng model
  │   ├── vit/         # Giang
  │   ├── mlp_mixer/   # Thy
  │   ├── resmlp/      # Minh
  │   ├── gmlp/        # Cường
  │   └── convmixer/   # Tiên
  ├── data/            # Dataset utils, CIFAR-10 loader
  ├── configs/         # Config files cho 3 mức kích thước
  ├── train.py         # Script train chung
  ├── eval.py          # Script đánh giá
  └── results/         # Kết quả chạy
  ```
- [ ] **Viết sẵn code base dùng chung**:
  - CIFAR-10 data loader (train 45k / val 5k / test 10k)
  - Training loop chuẩn (AdamW, cosine schedule, augmentation)
  - Evaluation script (accuracy, loss logging)
  - Config template cho 3 mức params

#### 🧪 Contributor (gMLP)
- [ ] Đọc kỹ lại paper gMLP — đặc biệt phần **Spatial Gating Unit (SGU)**
- [ ] Sketch kiến trúc gMLP trên giấy/note:
  - `gMLP Block: Norm → Linear(d→d_ffn) → GeLU → SGU → Linear(d_ffn→d) → +skip`
  - `SGU: split(Z₁, Z₂) → Z₁ ⊙ f(Norm(Z₂))` với f = Linear(n×n), init W≈0, b=1

---

### Thứ Sáu 25/09

#### 🎩 Leader
- [ ] **Check-in với từng thành viên**:
  - Giang: Đã hiểu ViT architecture chưa? Patch embed + position embed + [CLS] token?
  - Thy: Đã hiểu MLP-Mixer token-mixing vs channel-mixing chưa?
  - Minh: Đã hiểu ResMLP linear layer + Affine transform chưa?
  - Tiên: Đã hiểu ConvMixer depthwise conv + pointwise conv chưa?
- [ ] **Chia sẻ code base** cho nhóm, hướng dẫn cách dùng

#### 🧪 Contributor
- [ ] **Bắt đầu code gMLP** cho CIFAR-10 (patch 4×4):
  - Implement `SpatialGatingUnit` class
  - Implement `gMLPBlock` class  
  - Implement `gMLP` model (stack blocks + patch embed + classifier)

---

### Thứ Bảy–Chủ Nhật 26–28/09 — **Deadline nộp Tuần 1**

#### 🎩 Leader
- [ ] **Review Paper Tracker** lần cuối — đảm bảo 9 papers đều có đủ thông tin
- [ ] **Nộp Outline + Paper Tracker** cho giảng viên
- [ ] Viết **summary email/message** tuần 1 cho nhóm

#### 🧪 Contributor
- [ ] Hoàn thành code gMLP v1 (chưa cần chạy, đảm bảo forward pass đúng)

---

## 📅 TUẦN 2 — Cài đặt, chạy thử (29/09 – 05/10)

### Thứ Hai–Thứ Ba 29–30/09

#### 🎩 Leader
- [ ] **Tính toán 3 mức params** cho cả 5 model — đảm bảo fair comparison:

| Mức | Target params | Cách điều chỉnh |
|-----|--------------|-----------------|
| Small (~1M) | ~1M | Giảm depth + width |
| Medium (~3M) | ~3M | Depth/width trung bình |
| Large (~8M) | ~8M | Tăng depth + width |

- [ ] Tạo **bảng config** thống nhất cho cả nhóm:
  ```python
  # Ví dụ config cho gMLP
  configs = {
      "small":  {"depth": 6,  "dim": 128, "ffn_dim": 256},
      "medium": {"depth": 12, "dim": 192, "ffn_dim": 384},
      "large":  {"depth": 18, "dim": 256, "ffn_dim": 512},
  }
  ```
- [ ] **Gửi config cho nhóm** + hướng dẫn cách đếm params (`sum(p.numel() for p in model.parameters())`)

#### 🧪 Contributor
- [ ] **Debug gMLP code** trên Colab
- [ ] Đảm bảo model khởi tạo đúng (SGU init W≈0, b=1)
- [ ] Verify forward pass: input shape → output shape

---

### Thứ Tư–Thứ Năm 01–02/10

#### 🧪 Contributor (Ưu tiên cao)
- [ ] **Chạy thử gMLP** vài epoch (5–10) trên Colab
- [ ] Ghi lại:
  - Thời gian mỗi epoch (phút)
  - Ước lượng tổng thời gian cho 50–100 epochs
  - GPU memory sử dụng
  - Loss có giảm đều không? (sanity check)

#### 🎩 Leader
- [ ] **Thu thập báo cáo chạy thử** từ mỗi thành viên
- [ ] Tạo bảng so sánh thời gian chạy:

| Model | Thời gian/epoch | GPU mem | Ước lượng 100 epochs |
|-------|----------------|---------|---------------------|
| ViT | ? | ? | ? |
| MLP-Mixer | ? | ? | ? |
| ResMLP | ? | ? | ? |
| gMLP | ? | ? | ? |
| ConvMixer | ? | ? | ? |

- [ ] **Quyết định số epoch cuối cùng** dựa trên thời gian thực tế (50 hay 100?)
- [ ] Nếu model nào quá lâu → điều chỉnh batch size hoặc config

---

### Thứ Sáu–Chủ Nhật 03–05/10 — **Deadline nộp Tuần 2**

#### 🎩 Leader
- [ ] **Review code** của mỗi thành viên (ít nhất kiểm tra output shape + param count)
- [ ] **Debug** nếu ai gặp khó khăn
- [ ] **Nộp kết quả chạy thử** + bảng so sánh thời gian + nhận xét ngắn

#### 🧪 Contributor
- [ ] Đảm bảo gMLP chạy được ở cả 3 mức kích thước
- [ ] Viết nhận xét ngắn về khó khăn gặp phải

---

## 📅 TUẦN 3 — Chạy thí nghiệm chính (06/10 – 12/10)

> [!WARNING]
> Tuần quan trọng nhất! Cần **30 lần chạy** tổng cộng (5 model × 3 mức × 2 seed). Phải quản lý chặt tiến độ.

### Thứ Hai–Thứ Tư 06–08/10

#### 🧪 Contributor (Ưu tiên cao nhất)
- [ ] **Chạy gMLP Small** — Seed 1 + Seed 2
- [ ] **Chạy gMLP Medium** — Seed 1 + Seed 2
- [ ] Log kết quả vào bảng (accuracy, loss, thời gian)

#### 🎩 Leader
- [ ] **Tạo Google Sheet tracking** tiến độ chạy:

| Model | Size | Seed 1 Acc | Seed 2 Acc | Mean | Std | Người | Status |
|-------|------|-----------|-----------|------|-----|-------|--------|
| gMLP | S | ? | ? | ? | ? | Cường | ⏳ |
| gMLP | M | ? | ? | ? | ? | Cường | ⏳ |
| ... | ... | ... | ... | ... | ... | ... | ... |

- [ ] **Nhắc nhở nhóm** bắt đầu chạy, báo cáo tiến độ hàng ngày

---

### Thứ Năm–Thứ Sáu 09–10/10

#### 🧪 Contributor
- [ ] **Chạy gMLP Large** — Seed 1 + Seed 2
- [ ] Ghi nhận tất cả kết quả vào tracking sheet

#### 🎩 Leader
- [ ] **Check tiến độ nhóm** — ai chạy xong, ai chưa?
- [ ] Hỗ trợ debug nếu ai bị lỗi (OOM, NaN loss, etc.)
- [ ] Bắt đầu **tổng hợp kết quả** có rồi

---

### Thứ Bảy–Chủ Nhật 10–12/10 — **Deadline nộp Tuần 3**

#### 🎩 Leader (Quan trọng!)
- [ ] **Tổng hợp bảng kết quả cuối cùng** cho cả 5 model × 3 mức
- [ ] **Vẽ biểu đồ so sánh**:
  1. **Bar chart**: Accuracy vs Model ở mỗi mức params
  2. **Line chart**: Accuracy vs Params cho từng model
  3. **Ablation**: Accuracy khi bỏ token mixing (Thy thực hiện, Cường tổng hợp)
- [ ] **Nộp**: bảng + biểu đồ + nhận xét sơ bộ

> [!TIP]
> Yêu cầu **Thy** làm thêm bản MLP-Mixer bỏ token mixing từ sớm (đầu tuần 3) để kịp có kết quả.

---

## 📅 TUẦN 4 — Phân tích, viết báo cáo (13/10 – 19/10)

### Thứ Hai–Thứ Ba 13–14/10

#### 🧪 Contributor → Chuyển sang Writer
- [ ] **Viết phần Results** của báo cáo:
  - Trình bày bảng kết quả chính (5 model × 3 mức)
  - Phân tích: model nào thắng ở mức nào? Tại sao?
  - Kết quả ablation (bỏ token mixing)

#### 🎩 Leader
- [ ] **Giao deadline viết** cho từng người:
  - Tiên: Phần Mixer & ablation → **xong trước 15/10**
  - Minh: Phần so sánh kết quả → **xong trước 15/10**
  - Thy: Phần hạn chế → **xong trước 16/10**
  - Giang: Biểu đồ cuối + hướng dẫn chạy code → **xong trước 16/10**
- [ ] Tạo **template báo cáo** với outline:
  ```
  1. Giới thiệu (Cường)
  2. Related Work / Tổng quan papers (Tiên)
  3. Phương pháp — 5 kiến trúc (cả nhóm)
  4. Thiết kế thí nghiệm (Cường)
  5. Kết quả (Cường)
  6. So sánh & phân tích (Minh)
  7. Ablation study (Tiên)
  8. Hạn chế (Thy)
  9. Kết luận (Cường)
  10. Hướng dẫn reproduce (Giang)
  ```

---

### Thứ Tư–Thứ Năm 15–16/10

#### 🧪 Writer
- [ ] **Viết phần Kết luận**:
  - Trả lời 3 câu hỏi nghiên cứu
  - Self-attention có thực sự cần thiết không?
  - Kết quả phụ thuộc kiến trúc hay kích thước/dữ liệu?

#### 🎩 Leader
- [ ] **Review** bài viết của Tiên, Minh (đã nộp)
- [ ] **Feedback + yêu cầu chỉnh sửa** nếu cần
- [ ] Bắt đầu **ghép các phần** thành báo cáo hoàn chỉnh

---

### Thứ Sáu 17/10 — Họp chốt báo cáo

#### 🎩 Leader
- [ ] **Tổ chức họp nhóm** (online/offline):
  - Review toàn bộ báo cáo
  - Kiểm tra consistency (số liệu, biểu đồ, bảng)
  - Chốt version cuối
- [ ] **Giao việc sửa cuối** nếu cần

---

### Thứ Bảy–Chủ Nhật 18–19/10 — **DEADLINE CUỐI**

#### 🎩 Leader
- [ ] **Final review** toàn bộ:
  - [ ] Báo cáo: format, chính tả, logic
  - [ ] Code: chạy được, có README
  - [ ] Paper Tracker: hoàn chỉnh
- [ ] **NỘP**: Báo cáo cuối + Code + Paper Tracker hoàn chỉnh

---

## ⚡ Checklist Hành Động Ngay Hôm Nay (24/09)

> [!CAUTION]
> Đây là những việc bạn nên làm **ngay hôm nay** với tư cách leader:

- [ ] 🎩 Gửi message tổng kết Tuần 1 cho nhóm
- [ ] 🎩 Tạo shared folder/repo với cấu trúc thống nhất
- [ ] 🎩 Viết code base dùng chung (data loader, training loop, config)
- [ ] 🧪 Đọc kỹ paper gMLP, sketch kiến trúc SGU
- [ ] 🎩 Xác nhận mỗi thành viên đã đọc xong paper phụ trách

---

## 🔑 Nguyên Tắc Leader

1. **Đi trước 1 bước**: Code base dùng chung phải xong trước khi nhóm bắt đầu code model
2. **Deadline nội bộ sớm hơn**: Đặt deadline nội bộ **trước deadline thật 1 ngày** để có buffer
3. **Ghi lại mọi thay đổi**: Nếu config khác so với plan ban đầu → ghi lại lý do
4. **Check-in hàng ngày trong Tuần 3**: Tuần chạy thí nghiệm là quan trọng nhất
5. **Quality gate**: Không merge kết quả nào chưa được verify (kiểm tra param count, loss curve hợp lý)
