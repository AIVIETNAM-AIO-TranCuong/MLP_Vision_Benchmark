# 🚀 HƯỚNG DẪN TỪ A-Z: CÁCH CODE VÀ TRAIN MÔ HÌNH CHO TEAM MD4

Mình đã setup xong toàn bộ code base chung cho nhóm. Nhiệm vụ của mọi người bây giờ rất đơn giản: **Chỉ cần viết đúng kiến trúc mô hình của mình, tự training, rồi nộp lại kết quả cho mình.**

Mọi người không cần quan tâm đến việc tải dữ liệu CIFAR-10, không cần lo viết vòng lặp huấn luyện. Cứ làm đúng 3 Giai đoạn dưới đây là xong!

---

## 💻 GIAI ĐOẠN 1: CODE MÔ HÌNH (Làm trên máy tính của bạn)

Bất kể tí nữa bạn định train bằng máy cá nhân hay mượn Google Colab, thì bước gõ code này **BẮT BUỘC** phải làm trên máy tính cá nhân của bạn (dùng VSCode, PyCharm...).

### Bước 1: Viết file code của bạn
- Bạn hãy vào thư mục `models/`, tạo một file mới mang tên mô hình của bạn (Ví dụ anh Giang làm ViT thì tạo file `models/vit.py`).
- Viết class mô hình bằng thư viện PyTorch. Nhớ kỹ 2 điều: 
  - Đầu vào (Input) phải nhận ảnh kích thước `(Batch, 3, 32, 32)`.
  - Đầu ra (Output) phải trả về `(Batch, 10)`.
- Cuối file, hãy viết một hàm có tên `build_ten_mo_hinh(config)` (`build_vit`),để hệ thống có thể gọi ra dùng (có thể mở file `gmlp.py` của mình để tham khảo phần `build_gmlp`)

### Bước 2: Báo cho hệ thống biết mô hình của bạn đã ra đời
- Bạn mở file `models/__init__.py` ra.
- Tìm dòng `MODEL_REGISTRY`. Khai báo tên mô hình và tên cái hàm `build_...` mà bạn vừa viết ở Bước 1 vào đó.

### Bước 3: Đặt thông số (Parameters)
Mô hình của nhóm mình phải chia làm 3 bản: Nhỏ (Small), Vừa (Medium), Lớn (Large).
- Bạn mở file `configs/default.py` ra.
- Tìm đến khối cấu hình tên mô hình của bạn. Chỉnh sửa các con số (số layer, kích thước channel...) sao cho nó đạt đúng 3 mốc:
  - **Bản Small:** Cỡ ~1 triệu tham số
  - **Bản Medium:** Cỡ ~3 triệu tham số
  - **Bản Large:** Cỡ ~8 triệu tham số

### Bước 4: Chạy thử.
- Bạn mở Terminal (Cửa sổ lệnh) của máy tính lên, gõ: `python -m models.ten_model_cua_ban`
- Nếu Terminal in ra chữ "Forward pass OK" và số tham số giống hệt như bạn tính toán thì đã code xong. Tiếp theo, nhảy sang Giai đoạn 2.

---

## 🏃‍♂️ GIAI ĐOẠN 2: TỰ CHỌN NỀN TẢNG TRAIN MÔ HÌNH

Đã có code hoàn chỉnh trong máy tính, giờ bạn hãy tự đánh giá xem máy tính của mình mạnh hay yếu để chọn **1 trong 3 Option** dưới đây.

### 🔴 OPTION A: MÁY BẠN CÓ CARD ĐỒ HỌA MẠNH (Ví dụ: NVIDIA RTX 3060, 4070...)

1. Mở Terminal, gõ lệnh cài đặt 1 lần duy nhất: `pip install -r requirements.txt`.
2. Gõ lệnh Train: `python train.py --model tên_mô_hình --size small --seed 42` (nếu muốn train bản to hơn thì thay chữ `small` thành `medium` hoặc `large`).
3. Đợi train xong, kết quả đã tự động lưu vào thư mục `results/` trên máy tính của bạn.

---

### 🔵 OPTION B: MÁY BẠN YẾU -> HÃY MƯỢN GOOGLE COLAB CỦA GOOGLE
1. **Đưa code gg drive:** Đẩy toàn bộ thư mục `MD4_conquer` của bạn lên **Google Drive cá nhân của bạn**.
2. **Mở Colab:** Lên Google Drive, click đúp vào file `MD4_AutoTrain.ipynb`.
3. **Bật GPU:** Nhìn lên thanh menu trên cùng, chọn **Runtime** -> **Change runtime type** -> Tại mục Hardware accelerator chọn **T4 GPU** -> Bấm **Save**.
4. **Kết nối Drive:** Run **Ô code số 1**. Nó hỏi xin quyền truy cập Drive thì cứ bấm "Cho phép".
5. **Chép Code:** Run **Ô code số 2**. *(Lưu ý: Nếu bạn vứt thư mục MD4 ở chỗ khác trong Drive, hãy tự sửa lại đường dẫn chữ đỏ `PROJECT_DRIVE_PATH` trong ô code này cho đúng nhé).*
6. **Bấm Train:** Kéo xuống **Ô code số 3**. Sẽ có mấy cái ô chọn chọn bằng chuột rất xịn. Bạn chọn tên Model của bạn, chọn Size, rồi run code.
7. **LƯU KẾT QUẢ (SIÊU QUAN TRỌNG):** Sau khi nó chạy xong 100%, bạn **BẮT BUỘC** phải bấm Run ở **Ô code cuối cùng**. Ô này làm nhiệm vụ chép file kết quả từ máy ảo về lại cái Google Drive của bạn. Bạn mà lỡ tắt tab web khi chưa chạy ô này là mất sạch dữ liệu train nãy giờ.

---

### 🟢 OPTION C: GOOGLE COLAB BỊ LỖI -> DÙNG KAGGLE
*Nếu Google Colab chê bạn xài lố giờ không cho mượn GPU nữa, thì sang Kaggle mượn, tốc độ chạy còn nhanh gấp đôi Colab.*
1. **Nén Code:** Nén thư mục `MD4_conquer` ở dưới máy tính của bạn thành file `.zip`.
2. **Tạo Notebook mới:** Đăng nhập [Kaggle](https://www.kaggle.com/), bấm nút **Create** màu đen (góc trên bên trái) -> Chọn **New Notebook**.
3. **Up file Zip lên Kaggle:** Nhìn sang góc bên phải màn hình, tìm mục **Input**, bấm chữ **Upload Data**. Ném file Zip vào đó, đặt đại 1 cái tên (VD: `code-md4-cua-toi`) rồi bấm Create.
4. **Bật GPU:** Vẫn ở cột bên phải đó, tìm mục **Accelerator** -> Chọn **GPU T4 x2**. Bật thêm dòng **Internet** kế bên cho nó sáng lên.
5. **Chép Code ra nháp:** Bạn tạo 1 ô code (Code cell) trên màn hình Kaggle, copy 2 lệnh này ném vào và bấm Run:
   ```bash
   !cp -r /kaggle/input/code-md4-cua-toi/* /kaggle/working/
   %cd /kaggle/working/
   ```
6. **Bấm Train:** Bạn tạo thêm 1 ô Code trống ngay bên dưới, dán lệnh này vào và Run:
   ```bash
   !python train.py --model tên_mô_hình --size small --seed 42
   ```
7. **Lấy file về máy:** Train xong, nhìn cột bên phải phần **Output**. Bạn sẽ thấy thư mục `results/`. Hãy bấm nút Download (tải xuống) các file kết quả về máy tính cá nhân của bạn.

---

## 📊 GIAI ĐOẠN 3: NỘP KẾT QUẢ (Nhiệm vụ cuối cùng)

Bất kể bạn dùng option nào. Cứ kết thúc Giai đoạn 2 là bạn sẽ luôn thu được đúng 2 loại file kết quả:
- 📄 Dạng `.json` (Ví dụ: `vit_small_seed42.json`) -> Chứa biểu đồ điểm số.
- 🧱 Dạng `.pt` (Ví dụ: `vit_small_seed42_best.pt`) -> Chứa "bộ não" (trọng số) của AI nặng mấy chục MB.

**Nhiệm vụ của bạn:** Gửi tất cả các file này (của cả 3 bản Small, Medium, Large) cho Cường.
Cường sẽ có trách nhiệm gom tất cả file của cả 5 người lại và chạy lệnh để hệ thống tự động in ra bảng điểm tổng kết.
Chúc mọi người hoàn thành xuất sắc!
