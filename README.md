# MD4 — So sánh Kiến trúc MLP cho Image Classification

## 📋 Đề tài
So sánh 5 kiến trúc thay self-attention bằng MLP tĩnh trên CIFAR-10:
**ViT** | **MLP-Mixer** | **ResMLP** | **gMLP** | **ConvMixer**

## 🏗️ Cấu trúc Project
```
MD4_conquer/
├── docs/                    # Tài liệu quản lý dự án & tracking
│   ├── leader_action_plan.md
│   ├── Paper Tracker - Multilayer Perceptron.xlsx
│   └── WEEK01 - Outline and Plan.docx
├── data/
│   └── cifar10.py           # CIFAR-10 loader (45k/5k/10k split)
├── models/
│   ├── __init__.py          # Model registry
│   ├── gmlp.py              # 🔲 gMLP (Cường)
│   ├── vit.py               # 🔲 ViT (Giang)
│   ├── mlp_mixer.py         # 🔲 MLP-Mixer (Thy)
│   ├── resmlp.py            # 🔲 ResMLP (Minh)
│   └── convmixer.py         # 🔲 ConvMixer (Tiên)
├── configs/
│   └── default.py           # Tất cả configs
├── train.py                 # Script train chung
├── evaluate.py              # Script đánh giá + tổng hợp
├── utils.py                 # Utilities dùng chung
├── results/                 # Kết quả chạy (JSON + checkpoints)
└── requirements.txt
```

## 🚀 Cách sử dụng

### 1. Cài đặt
```bash
pip install -r requirements.txt
```

### 2. Train model
```bash
# Train gMLP small, seed 42
python train.py --model gmlp --size small --seed 42

# Train gMLP medium, 50 epochs
python train.py --model gmlp --size medium --seed 42 --epochs 50

# Train với 50% dữ liệu (câu hỏi nghiên cứu 2)
python train.py --model gmlp --size small --seed 42 --train_subset 0.5

# Trên Google Colab (giảm num_workers)
python train.py --model gmlp --size small --seed 42 --num_workers 0
```

### 3. Đánh giá
```bash
# Đánh giá 1 checkpoint
python evaluate.py --checkpoint results/checkpoints/gmlp_small_seed42_best.pt \
                   --model gmlp --size small

# Tổng hợp tất cả kết quả
python evaluate.py --results_dir results/
```

### 4. Verify model (đếm params + test forward pass)
```bash
python -m models.gmlp
```

## 👥 Phân công

| Thành viên | Model | File cần tạo |
|-----------|-------|-------------|
| Cường | gMLP | `models/gmlp.py` |
| Giang | ViT | `models/vit.py` |
| Thy | MLP-Mixer | `models/mlp_mixer.py` |
| Minh | ResMLP | `models/resmlp.py` |
| Tiên | ConvMixer | `models/convmixer.py` |

## 📝 Hướng dẫn thêm Model mới

1. Tạo file `models/your_model.py`
2. Implement model class với `forward(x)`: input `(B, 3, 32, 32)` → output `(B, 10)`
3. Tạo factory function `build_your_model(config: dict) -> nn.Module`
4. Đăng ký trong `models/__init__.py`:
   ```python
   from models.your_model import build_your_model
   MODEL_REGISTRY["your_model"] = build_your_model
   ```
5. Thêm config vào `configs/default.py` → `MODEL_CONFIGS["your_model"]`
6. Train: `python train.py --model your_model --size small --seed 42`

## ⚙️ Config chung
- **Optimizer**: AdamW (lr=1e-3, weight_decay=0.05)
- **Scheduler**: Cosine + Warmup (5 epochs)
- **Augmentation**: RandomCrop(32, pad=4) + HFlip
- **Label smoothing**: 0.1
- **Epochs**: 100
- **Batch size**: 128
- **Patch size**: 4×4 (→ 64 patches cho CIFAR-10)
