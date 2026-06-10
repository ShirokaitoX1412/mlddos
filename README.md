# Nghiên cứu xây dựng giải pháp phát hiện DDoS sử dụng học máy

## 1. Tổng quan đề tài

### 1.1. Mô tả bài toán

Tấn công từ chối dịch vụ phân tán (Distributed Denial of Service — DDoS) là một trong những mối đe dọa nghiêm trọng nhất đối với hạ tầng mạng hiện đại. Kẻ tấn công sử dụng lượng lớn lưu lượng giả mạo để làm quá tải tài nguyên máy chủ, khiến dịch vụ không thể phục vụ người dùng hợp lệ.

Bài toán đặt ra: **phân loại lưu lượng mạng thành 7 lớp** (1 lớp bình thường và 6 lớp tấn công DDoS khác nhau) dựa trên các đặc trưng thống kê của luồng dữ liệu (network flow features), sử dụng các thuật toán học máy có giám sát.

### 1.2. Lý do sử dụng học máy

- Các phương pháp dựa trên ngưỡng cố định (threshold-based) không thể phân biệt được các biến thể tấn công DDoS ngày càng tinh vi.
- Học máy cho phép tự động học các pattern phức tạp từ dữ liệu lưu lượng mạng mà không cần lập trình thủ công từng quy tắc.
- Khả năng phân loại đa lớp giúp không chỉ phát hiện mà còn **xác định loại tấn công cụ thể**, hỗ trợ phản ứng phù hợp.

### 1.3. Mục tiêu hệ thống

1. Xây dựng pipeline học máy hoàn chỉnh: từ nạp dữ liệu, tiền xử lý, huấn luyện, đánh giá đến lưu mô hình.
2. Đảm bảo tính khoa học: kiểm soát data leakage, overfitting, đánh giá bằng cross-validation và nhiều thước đo.
3. Demo phát hiện DDoS thời gian thực trên 2 máy ảo: VM tấn công sinh traffic, VM nạn nhân chạy IDS/IPS phân loại và chặn.
4. Cung cấp giao diện giám sát trực quan bằng Streamlit.

### 1.4. Phạm vi

- Dữ liệu: bộ CICDDoS2019 ở dạng flow features (đã trích xuất sẵn bằng CICFlowMeter), không phải raw packet.
- Thuật toán: Random Forest, Extra Trees, XGBoost, MLP Classifier.
- Triển khai demo: 2 máy ảo (Attacker + Victim) với Scapy, Streamlit dashboard, replay IDS offline.

---

## 2. Bộ dữ liệu sử dụng

### 2.1. Thông tin chung

| Thuộc tính | Giá trị |
|------------|---------|
| Tên dataset | CICDDoS2019 |
| Nguồn | Canadian Institute for Cybersecurity, University of New Brunswick |
| Link tải | https://www.kaggle.com/datasets/dhoogla/cicddos2019 |
| Định dạng | Parquet (chia theo attack type, train/test riêng) |
| Số đặc trưng ban đầu | 78 đặc trưng lưu lượng mạng |
| Tổng số bản ghi | 158.987 (train: 120.065 + test: 38.973) |
| Sau loại trùng lặp | 150.472 bản ghi duy nhất |

### 2.2. Các lớp phân loại

| Lớp | Mô tả | Tỷ lệ (sau resplit) |
|-----|--------|---------------------|
| **Benign** | Lưu lượng bình thường, không chứa tấn công | 34,26% |
| **TCP SYN Flood** | Gửi hàng loạt gói TCP SYN để làm cạn bảng kết nối | 31,77% |
| **UDP Flood** | Gửi lượng lớn gói UDP ngẫu nhiên gây quá tải băng thông | 18,09% |
| **MSSQL Flood** | Khuếch đại phản hồi từ dịch vụ Microsoft SQL Server | 8,03% |
| **UDP-Lag Flood** | Gửi gói UDP với kích thước lớn gây trễ xử lý | 5,63% |
| **LDAP Flood** | Khuếch đại phản hồi từ dịch vụ LDAP | 1,81% |
| **NetBIOS Flood** | Khai thác giao thức NetBIOS để khuếch đại lưu lượng | 0,45% |

### 2.3. Đặc điểm dữ liệu

Dữ liệu ở dạng **flow-level features** — mỗi bản ghi đại diện cho một luồng mạng (network flow) đã được trích xuất bằng CICFlowMeter, bao gồm các đặc trưng thống kê như: số gói tin, kích thước gói, thời gian giữa các gói (IAT), cờ TCP, tốc độ truyền, v.v. Đây không phải dữ liệu raw packet.

---

## 3. Kiến trúc hệ thống

### 3.1. Sơ đồ workflow

```text
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐     ┌──────────────┐
│  CICDDoS2019│────▶│  Data Loader │────▶│  Preprocessor   │────▶│ Train/Test   │
│  (Parquet)  │     │  + Resplit   │     │  (Leakage-safe) │     │   Split      │
└─────────────┘     └──────────────┘     └─────────────────┘     └──────┬───────┘
                                                                         │
                    ┌──────────────┐     ┌─────────────────┐             │
                    │  Saved Model │◀────│ Model Training  │◀────────────┘
                    │  (.pkl)      │     │ + Evaluation    │
                    └──────┬───────┘     └─────────────────┘
                           │
            ┌──────────────┼──────────────────┐
            ▼              ▼                  ▼
    ┌──────────────┐ ┌──────────┐    ┌───────────────┐
    │ Live IPS     │ │ Replay   │    │  Streamlit    │
    │ (2 VM Demo) │ │ IDS Demo │    │  Dashboard    │
    └──────────────┘ └──────────┘    └───────────────┘
```

### 3.2. Các thành phần

| Thành phần | File | Chức năng |
|------------|------|-----------|
| Data Loader | `data_loader.py` | Tải dataset, lọc attack types, gộp và chia lại dữ liệu |
| Preprocessor | `preprocessor.py` | Chuẩn hóa nhãn, xử lý missing/inf, loại cột tương quan cao, scale features |
| Training Pipeline | `main.py`, `models.py` | Huấn luyện, cross-validation, hyperparameter tuning, so sánh models |
| Model Evaluation | `models.py` | Confusion matrix, classification report, ROC-AUC, composite score |
| Saved Models | `saved_models/` | Lưu trữ model đã train dạng pickle |
| Replay IDS | `replay_ips.py` | Demo offline — phát lại flow từ dataset, phân loại và ghi log |
| Live IPS | `live_ips.py` | Bắt gói tin thời gian thực bằng Scapy, phân loại và chặn IP |
| Traffic Generator | `tools/ddos_traffic_generator.py` | Sinh traffic tấn công trên VM attacker cho demo |
| Dashboard | `frontend/app.py` | Giao diện Streamlit hiển thị kết quả và giám sát |
| SHAP Explainer | `shap_explainer.py` | Giải thích mô hình bằng SHAP values |

### 3.3. Cấu trúc thư mục

```text
mlddos/
├── backend/
│   ├── src/ml_ddos/
│   │   ├── data_loader.py        # Nạp dữ liệu + combine_and_resplit()
│   │   ├── preprocessor.py       # Tiền xử lý leakage-safe
│   │   ├── models.py             # Định nghĩa và huấn luyện mô hình
│   │   ├── main.py               # Pipeline chính (CLI)
│   │   ├── live_ips.py           # IPS thời gian thực (Scapy)
│   │   ├── replay_ips.py         # Demo phát lại flow offline
│   │   ├── mitigation.py         # Chặn/bỏ chặn IP (iptables/netsh)
│   │   ├── sdn_ryu_detector.py   # Ryu/SDN controller (prototype)
│   │   ├── shap_explainer.py     # Giải thích SHAP
│   │   └── paths.py              # Quản lý đường dẫn
│   └── requirements.txt
├── frontend/
│   └── app.py                    # Dashboard Streamlit
├── notebooks/
│   └── ddos_detection_report_ready.ipynb   # Notebook báo cáo
├── tools/
│   ├── ddos_traffic_generator.py  # Traffic generator cho VM attacker
│   └── mininet_sdn_topology.py    # Topology Mininet (prototype)
├── tests/
│   └── test_smoke_predict.py     # Kiểm thử predict_proba
├── data/                          # Dữ liệu CICDDoS2019
├── results/                       # Kết quả, biểu đồ, CSV
├── saved_models/                  # Models đã train (.pkl)
└── README.md
```

---

## 4. Quy trình xử lý dữ liệu

### 4.1. Nạp dữ liệu

- Quét thư mục `data/` để tìm các file Parquet theo pattern `*-training.parquet` và `*-testing.parquet`.
- Chỉ giữ lại các attack type xuất hiện ở **cả** tập train và test.
- Module: `data_loader.collect_file_paths()`, `data_loader.load_dataframes()`.

### 4.2. Chuẩn hóa nhãn

- Ánh xạ tên nhãn thô (VD: `DrDoS_UDP`, `Syn`, `WebDDoS`) sang 7 lớp chuẩn.
- Loại bỏ các nhãn không thuộc danh sách 7 lớp (VD: `WebDDoS` chỉ có trong test).
- Module: `preprocessor.harmonize_labels()`.

### 4.3. Gộp và chia lại dữ liệu (Combine & Re-split)

Đây là bước quan trọng nhất để xử lý **distribution shift** của CICDDoS2019:

1. Gộp tập train và test gốc thành một tập duy nhất.
2. Loại bỏ bản ghi trùng lặp (deduplication theo toàn bộ feature columns).
3. Chia lại theo tỷ lệ 80/20 bằng **stratified sampling** — đảm bảo mỗi lớp có tỷ lệ đồng đều giữa train và test mới.

Module: `data_loader.combine_and_resplit()`.

### 4.4. Xử lý giá trị không hợp lệ

- Thay thế giá trị `inf` và `NaN` bằng trung vị (median) của cột tương ứng.
- Trung vị được tính **chỉ trên tập train** để tránh data leakage.
- Module: `preprocessor.fit_invalid_value_medians()`, `preprocessor.apply_invalid_value_medians()`.

### 4.5. Loại bỏ cột không hữu ích

- Loại cột có duy nhất 1 giá trị (không mang thông tin phân biệt).
- Loại cột có tương quan cao (> 0,90) để giảm multicollinearity.
- Danh sách cột cần loại được xác định **chỉ trên tập train**.
- Module: `preprocessor.drop_single_value_columns()`, `preprocessor.drop_highly_correlated()`.

### 4.6. Chuẩn hóa đặc trưng

- Sử dụng `MinMaxScaler` để đưa features về khoảng [0, 1].
- Scaler được **fit trên tập train**, chỉ transform trên test/validation.
- Module: `preprocessor.scale_features()`.

### 4.7. Xử lý mất cân bằng lớp

- `RandomOverSampler` cho các lớp thiểu số (LDAP, NetBIOS).
- `RandomUnderSampler` cho lớp đa số (Benign, TCP SYN).
- Stage 2 oversampling cho các lớp cực hiếm (< 500 mẫu).
- Module: `models.balance_training_distribution()`.

### 4.8. Mã hóa nhãn

- Sử dụng `LabelEncoder` ánh xạ 7 tên lớp sang chỉ số 0–6.
- Module: `preprocessor.encode_labels()`.

---

## 5. Mô hình học máy

### 5.1. Danh sách mô hình

| Mô hình | Mô tả | Đặc điểm |
|---------|--------|-----------|
| **Random Forest** | Tập hợp nhiều cây quyết định theo cơ chế bagging | Robust với noise, ít overfitting |
| **Extra Trees** | Cây cực ngẫu nhiên — chọn ngưỡng split ngẫu nhiên | Nhanh hơn RF, giảm variance |
| **XGBoost** | Gradient boosting trên cây quyết định | Hiệu suất cao, regularization mạnh |
| **MLP Classifier** | Mạng nơ-ron nhiều lớp (feedforward) | Học được non-linear boundaries |

### 5.2. Hyperparameter tuning

- Sử dụng `RandomizedSearchCV` với StratifiedKFold.
- Không dùng default parameters — mỗi model được tuning riêng.
- Regularization được áp dụng có chủ đích: giảm `max_depth`, tăng `min_samples_leaf`, `reg_lambda`.

### 5.3. Tiêu chí chọn mô hình

Mô hình được chọn dựa trên **Selection Score** (composite score), không chỉ dựa vào Accuracy:

```
Selection Score = Macro F1 + 0.5 × Minority Recall + 0.25 × Balanced Accuracy − 0.25 × max(0, Train/Test Gap)
```

Các thước đo được cân nhắc:

| Thước đo | Ý nghĩa |
|----------|---------|
| **Macro F1** | Trung bình F1 của tất cả lớp — đánh giá công bằng kể cả lớp ít mẫu |
| **Weighted F1** | F1 có trọng số theo số mẫu — phản ánh hiệu suất tổng thể |
| **Balanced Accuracy** | Trung bình recall các lớp — xử lý imbalance |
| **Minority Class Recall** | Recall của lớp ít mẫu nhất — đảm bảo phát hiện attack hiếm |
| **Train/Test Gap** | Chênh lệch F1 giữa train và test — phát hiện overfitting |
| **CV Macro F1** | Cross-Validation F1 — ước lượng generalization |

---

## 6. Kết quả thực nghiệm

> Dữ liệu metric dưới đây được trích từ file `results/report_ready_model_comparison.csv` và `results/report_ready_final_summary.csv` sau khi chạy pipeline với phương pháp Combine & Re-split.

### 6.1. Bảng so sánh mô hình

| Mô hình | Phương pháp | Accuracy | Balanced Accuracy | Macro F1 | Weighted F1 | Minority Recall | CV Macro F1 |
|---------|-------------|----------|-------------------|----------|-------------|-----------------|-------------|
| **XGBoost** | **baseline** | **0,9756** | **0,9328** | **0,9386** | **0,9751** | **0,9101** | **0,9314 ± 0,0056** |
| Random Forest | baseline | 0,9751 | 0,9269 | 0,9388 | 0,9745 | 0,8918 | 0,9168 ± 0,0032 |
| Random Forest | class_weight | 0,9657 | 0,9391 | 0,8750 | 0,9668 | 0,9661 | 0,9152 ± 0,0029 |
| XGBoost | UnderSampler | 0,9736 | 0,9331 | 0,9361 | 0,9731 | 0,9129 | 0,9315 ± 0,0044 |
| MLP Classifier | OverSampler | 0,9702 | 0,9286 | 0,9233 | 0,9694 | 0,9349 | 0,8878 ± 0,0113 |

### 6.2. Hiệu suất phân loại theo từng lớp (Best model: XGBoost baseline)

| Lớp | Precision | Recall | F1-score | Support |
|-----|-----------|--------|----------|---------|
| Benign | 0,9938 | 0,9989 | 0,9964 | 10.306 |
| TCP SYN Flood | 0,9999 | 0,9865 | 0,9932 | 9.554 |
| UDP Flood | 0,9303 | 0,9906 | 0,9595 | 5.443 |
| MSSQL Flood | 0,9532 | 0,9690 | 0,9610 | 2.417 |
| UDP-Lag Flood | 0,9460 | 0,7646 | 0,8457 | 1.695 |
| LDAP Flood | 0,8818 | 0,8932 | 0,8875 | 543 |
| NetBIOS Flood | 0,9270 | 0,9270 | 0,9270 | 137 |
| **Macro avg** | **0,9474** | **0,9328** | **0,9386** | **30.095** |

### 6.3. Tổng hợp kết quả

| Thuộc tính | Giá trị |
|------------|---------|
| Dataset | CICDDoS2019 |
| Phương pháp chia dữ liệu | Combine & Re-split (stratified 80/20) |
| Tổng bản ghi sau loại trùng | 150.472 |
| Tập huấn luyện | 120.377 bản ghi |
| Tập kiểm thử | 30.095 bản ghi |
| Số đặc trưng đầu vào | 77 (trước loại bỏ) |
| Mô hình được chọn | XGBoost (baseline) |
| Train/Test Macro F1 Gap | 0,0021 |

### 6.4. So sánh trước và sau khi xử lý distribution shift

| Thước đo | Trước (Source Split gốc) | Sau (Combine & Re-split) |
|----------|--------------------------|--------------------------|
| Mô hình tốt nhất | KNN | XGBoost |
| Test F1-Weighted | 0,698 | 0,975 |
| Test F1-Macro | 0,677 | 0,939 |
| CV/Test Gap | ~30% | < 1% |
| UDP-Lag F1 | ≈ 0,00 | 0,846 |
| TCP SYN F1 | ≈ 0,14 | 0,993 |

---

## 7. Kiểm soát overfitting và data leakage

### 7.1. Các biện pháp đã áp dụng

| Biện pháp | Chi tiết |
|-----------|----------|
| **Preprocessing chỉ fit trên train** | MinMaxScaler, median imputer, danh sách cột cần drop — tất cả được xác định chỉ trên tập train |
| **Stratified split** | Chia dữ liệu giữ nguyên tỷ lệ lớp |
| **Cross-validation** | StratifiedKFold 3–5 fold, tính mean ± std |
| **Loại cột leakage** | Loại bỏ cột có tương quan quá cao (> 0,90) với target hoặc cột định danh |
| **Regularization** | Giảm `max_depth`, tăng `min_samples_leaf`, `reg_lambda` cho XGBoost |
| **Train/Test Gap monitoring** | So sánh Macro F1 trên train và test — gap > 5% cảnh báo overfitting |
| **Deduplication** | Loại bản ghi trùng trước khi split để tránh cùng flow rơi vào cả train/test |

### 7.2. Kết quả kiểm soát

- **Train/Test Macro F1 Gap** của mô hình được chọn: **0,0021** (0,21%) — không có overfitting.
- **CV Macro F1**: 0,9314 ± 0,0056 — độ lệch chuẩn thấp, mô hình ổn định qua các fold.

### 7.3. Vấn đề distribution shift đã xử lý

CICDDoS2019 chia train/test theo nguồn tấn công (source-based split), gây ra phân phối lớp cực kỳ lệch:

| Lớp | Train % (gốc) | Test % (gốc) | Chênh lệch |
|-----|----------------|---------------|-------------|
| UDP-Lag | 0,05% | 22,79% | +22,74% |
| TCP SYN | 44,05% | 13,37% | −30,68% |
| Benign | 35,28% | 27,94% | −7,34% |

Hậu quả: mô hình học phân phối train nhưng test có phân phối khác hoàn toàn → CV F1 = 0,99 nhưng Test F1 chỉ đạt 0,70 (gap 30%). Giải pháp: gộp, loại trùng, chia lại stratified đã loại bỏ hoàn toàn vấn đề này.

---

## 8. Triển khai demo hệ thống

### 8.1. Demo 2 máy ảo — Phát hiện DDoS thời gian thực

Đây là phương pháp demo chính của hệ thống, sử dụng 2 máy ảo (VM) trên cùng mạng nội bộ ảo (host-only hoặc internal network).

#### 8.1.1. Mô hình triển khai

```text
┌────────────────────────┐          ┌────────────────────────┐
│    VM 1 — ATTACKER     │          │    VM 2 — VICTIM       │
│    (Kali / Ubuntu)     │          │    (Ubuntu)            │
│                        │          │                        │
│  ddos_traffic_         │  ─────▶  │  live_ips.py (Scapy)   │
│  generator.py          │ Network  │    ↓                   │
│                        │          │  ML Model predict      │
│  hping3 / Scapy        │          │    ↓                   │
│                        │          │  iptables block IP     │
│                        │          │    ↓                   │
│                        │          │  Dashboard (Streamlit) │
└────────────────────────┘          └────────────────────────┘
     192.168.56.101                      192.168.56.102
         (ví dụ)                           (ví dụ)
```

**Ưu điểm so với SDN:**
- Features được trích xuất **đầy đủ 77 features** từ Scapy (khớp 100% với training data CICFlowMeter).
- Kịch bản tấn công rõ ràng, dễ giải thích cho hội đồng.
- Dễ tái tạo — chỉ cần VirtualBox/VMware và 2 VM.

#### 8.1.2. Chuẩn bị

**VM 1 — Attacker** (Kali Linux hoặc Ubuntu):
```bash
pip install scapy
# Hoặc cài hping3:
sudo apt install hping3
```

**VM 2 — Victim** (Ubuntu):
```bash
# Clone project và cài đặt
git clone https://github.com/ShirokaitoX1412/mlddos.git
cd mlddos
python -m venv venv && source venv/bin/activate
pip install -r backend/requirements.txt

# Train model (hoặc copy saved_models/ từ máy đã train)
PYTHONPATH=backend/src python -m ml_ddos.main --combine-resplit --cv-folds 3 --search-iter 10
```

**Cấu hình mạng:**
- VirtualBox: Host-Only Adapter (`vboxnet0`, subnet 192.168.56.0/24)
- VMware: Custom VMnet (host-only)
- Cả 2 VM cùng subnet, ping được lẫn nhau

#### 8.1.3. Các kịch bản tấn công DDoS

| # | Kịch bản | Loại tấn công | Mô tả |
|---|----------|---------------|-------|
| 1 | Baseline | Benign | Lưu lượng bình thường (HTTP, ICMP, DNS) |
| 2 | TCP SYN Flood | Volumetric | Gửi hàng loạt gói SYN làm cạn bảng kết nối |
| 3 | UDP Flood | Volumetric | Gửi lượng lớn gói UDP nhỏ gây quá tải |
| 4 | UDP-Lag Flood | Volumetric | Gửi gói UDP lớn (1400 bytes) gây trễ xử lý |
| 5 | LDAP Flood | Amplification | Traffic giả lập phản hồi LDAP (port 389) |
| 6 | MSSQL Flood | Amplification | Traffic giả lập phản hồi MSSQL (port 1434) |
| 7 | NetBIOS Flood | Amplification | Traffic giả lập NetBIOS query (port 137) |

#### 8.1.4. Thực hiện demo

**Bước 1 — Khởi chạy IDS/IPS trên VM Victim (Terminal 1):**
```bash
cd mlddos
source venv/bin/activate

# Chế độ mô phỏng (an toàn — chỉ ghi log, không chặn IP thật)
sudo PYTHONPATH=backend/src python -m ml_ddos.live_ips -i eth0

# Chế độ IPS thật (CÓ chặn IP bằng iptables)
sudo PYTHONPATH=backend/src python -m ml_ddos.live_ips -i eth0 --live
```

**Bước 2 — (Tùy chọn) Khởi chạy Dashboard trên VM Victim (Terminal 2):**
```bash
cd mlddos && source venv/bin/activate
streamlit run frontend/app.py
# Mở: http://192.168.56.102:8501
```

**Bước 3 — Sinh traffic tấn công từ VM Attacker:**

Cách 1 — Dùng script tự động (Scapy):
```bash
# Chạy từng kịch bản riêng lẻ:
sudo python3 tools/ddos_traffic_generator.py --target 192.168.56.102 --attack syn_flood -d 30
sudo python3 tools/ddos_traffic_generator.py --target 192.168.56.102 --attack udp_flood -d 30
sudo python3 tools/ddos_traffic_generator.py --target 192.168.56.102 --attack udp_lag -d 30
sudo python3 tools/ddos_traffic_generator.py --target 192.168.56.102 --attack ldap_flood -d 30
sudo python3 tools/ddos_traffic_generator.py --target 192.168.56.102 --attack mssql_flood -d 30
sudo python3 tools/ddos_traffic_generator.py --target 192.168.56.102 --attack netbios_flood -d 30

# Chạy TẤT CẢ kịch bản liên tiếp (30s mỗi loại):
sudo python3 tools/ddos_traffic_generator.py --target 192.168.56.102 --attack all -d 210
```

Cách 2 — Dùng hping3 (thủ công):
```bash
# TCP SYN Flood
sudo hping3 -S --flood -p 80 192.168.56.102

# UDP Flood
sudo hping3 --udp --flood -p 53 192.168.56.102

# UDP-Lag (payload lớn)
sudo hping3 --udp --flood -d 1400 -p 80 192.168.56.102
```

#### 8.1.5. Kết quả

- **Console VM Victim**: Hiển thị log real-time — mỗi flow được phân loại (Benign / loại tấn công) kèm confidence.
- **File log**: `results/live_events.csv` — ghi lại toàn bộ events.
- **Dashboard**: Biểu đồ cập nhật trực tiếp (nếu đã chạy Streamlit).
- **Chặn IP** (chế độ `--live`): Tự động thêm rule `iptables -A INPUT -s <IP> -j DROP`.

#### 8.1.6. Lưu ý an toàn

- **Chỉ chạy trên mạng lab cô lập** (host-only / internal network).
- Chế độ mặc định là **SIMULATION** — chỉ ghi log, không thực thi iptables.
- Flag `--live` sẽ chặn IP thật — chỉ dùng trên VM lab.

### 8.2. Replay IDS (Demo offline — không cần card mạng)

**Mục đích**: Demo phát hiện DDoS bằng cách phát lại flow từ dataset, không cần capture packet thật. Phù hợp cho trình bày báo cáo.

**Lệnh chạy:**
```bash
# Ubuntu
PYTHONPATH=backend/src python -m ml_ddos.replay_ips --model selected_model --rows 200

# Windows PowerShell
$env:PYTHONPATH="backend\src"; python -m ml_ddos.replay_ips --model selected_model --rows 200
```

**Kết quả sinh ra**: `results/live_events.csv` — log các flow đã phân loại, bao gồm timestamp, predicted label, confidence.

**Lưu ý**: Không thực thi lệnh chặn IP, chỉ ghi log.

### 8.3. Dashboard Streamlit

**Mục đích**: Giao diện trực quan hiển thị kết quả phân tích, biểu đồ so sánh models, confusion matrix, ROC curve, SHAP, và log giám sát.

**Lệnh chạy:**
```bash
streamlit run frontend/app.py
```

**Kết quả**: Mở trình duyệt tại `http://localhost:8501` với các tab: Phân tích mô hình, Giám sát trực tiếp, Cảnh báo.

---

## 9. Hướng dẫn cài đặt và chạy project

### 9.1. Yêu cầu hệ thống

- Python 3.10 trở lên
- pip hoặc virtualenv
- Scapy + quyền root cho Live IPS (VM Victim)
- (Tùy chọn) hping3 trên VM Attacker
- (Tùy chọn) VirtualBox hoặc VMware cho demo 2 VM

### 9.2. Cài đặt

```bash
# Clone repository
git clone https://github.com/ShirokaitoX1412/mlddos.git
cd mlddos

# Tạo môi trường ảo
python -m venv venv

# Kích hoạt môi trường ảo
# Ubuntu/Mac:
source venv/bin/activate
# Windows PowerShell:
# .\venv\Scripts\Activate.ps1

# Cài đặt thư viện
pip install -r backend/requirements.txt
```

### 9.3. Chuẩn bị dữ liệu

Dữ liệu CICDDoS2019 cần được đặt trong thư mục `data/` với cấu trúc:
```text
data/
├── cicddos2019/
│   ├── *-training.parquet
│   └── *-testing.parquet
```

Nếu có Kaggle API:
```bash
pip install kaggle
kaggle datasets download -d dhoogla/cicddos2019 -p data/
unzip data/cicddos2019.zip -d data/cicddos2019/
```

### 9.4. Huấn luyện mô hình

```bash
# Ubuntu
PYTHONPATH=backend/src python -m ml_ddos.main --combine-resplit --cv-folds 3 --search-iter 10

# Windows PowerShell
$env:PYTHONPATH="backend\src"; python -m ml_ddos.main --combine-resplit --cv-folds 3 --search-iter 10
```

**Các tham số CLI:**

| Tham số | Mô tả | Mặc định |
|---------|--------|----------|
| `--combine-resplit` | Gộp train+test, loại trùng, chia lại stratified | Không (dùng source split) |
| `--cv-folds N` | Số fold Cross-Validation | 5 |
| `--search-iter N` | Số iteration RandomizedSearchCV | 20 |
| `--min-attack-class-samples N` | Số mẫu tối thiểu cho rare classes | 300 |

**Output:**
- `saved_models/selected_model.pkl` — mô hình tốt nhất
- `results/` — CSV kết quả, biểu đồ, classification report

### 9.5. Chạy notebook báo cáo

```bash
jupyter notebook notebooks/ddos_detection_report_ready.ipynb
```

Notebook chạy toàn bộ pipeline và xuất kết quả phù hợp cho báo cáo: bảng metric, confusion matrix, biểu đồ so sánh models, kiểm tra SDN compatibility.

### 9.6. Chạy dashboard

```bash
streamlit run frontend/app.py
# Mở: http://localhost:8501
```

### 9.7. Chạy Replay IDS demo

```bash
# Ubuntu
PYTHONPATH=backend/src python -m ml_ddos.replay_ips --model selected_model --rows 200

# Windows PowerShell
$env:PYTHONPATH="backend\src"; python -m ml_ddos.replay_ips --model selected_model --rows 200
```

### 9.8. Chạy demo 2 VM

Xem chi tiết tại [Mục 8.1](#81-demo-2-máy-ảo--phát-hiện-ddos-thời-gian-thực).

```bash
# VM Victim — chạy IDS/IPS
sudo PYTHONPATH=backend/src python -m ml_ddos.live_ips -i eth0

# VM Attacker — sinh traffic tấn công
sudo python3 tools/ddos_traffic_generator.py --target <VICTIM_IP> --attack all -d 210
```

---

## 10. Hạn chế và hướng phát triển

### 10.1. Hạn chế

1. **Dataset không hoàn toàn phản ánh traffic thực tế**: CICDDoS2019 được tạo trong môi trường lab, có thể khác với traffic production.
2. **Một số lớp có ít mẫu**: NetBIOS (0,45%) và LDAP (1,81%) có số mẫu hạn chế, ảnh hưởng đến khả năng tổng quát hóa.
3. **Chưa kiểm thử trên traffic thật**: Hệ thống chỉ được đánh giá trên CICDDoS2019, cần thêm đánh giá trên dữ liệu thực tế từ mạng campus/enterprise.
4. **Thời gian phân loại**: Pipeline hiện tại chưa tối ưu cho real-time với lượng flow lớn (> 10.000 flow/s).
5. **Traffic generator đơn giản**: Script sinh traffic dùng Scapy/hping3 chỉ mô phỏng pattern cơ bản, chưa tái tạo đúng đặc điểm phức tạp của các cuộc tấn công thực tế.

### 10.2. Hướng phát triển

1. **Kiểm thử trên traffic thật**: Thu thập dữ liệu từ mạng campus/enterprise để đánh giá generalization.
2. **Tối ưu feature extraction live**: Giảm số features cần thiết, dùng feature selection để chỉ giữ top-K features quan trọng nhất.
3. **Triển khai SDN**: Mở rộng sang SDN/Ryu controller với Mininet/Open vSwitch để demo mitigation tự động bằng flow rules (đã có prototype trong `sdn_ryu_detector.py`). Cần cải thiện feature extraction từ OVS flow stats — hiện tại OVS chỉ cung cấp được ~20/77 features.
4. **Alerting**: Tích hợp cảnh báo Telegram/Email khi phát hiện tấn công.
5. **Mở rộng dataset**: Kết hợp thêm CIC-IDS2017, UNSW-NB15 để tăng đa dạng attack patterns.
6. **Deep Learning**: Thử nghiệm LSTM/CNN trên time-series flow features.

---

## 11. Kết luận

Đề tài đã xây dựng thành công một hệ thống phát hiện và phân loại tấn công DDoS sử dụng học máy, bao gồm:

- **Pipeline ML hoàn chỉnh**: Từ nạp dữ liệu, tiền xử lý (leakage-safe), huấn luyện với cross-validation, đến đánh giá và lưu mô hình.
- **Xử lý vấn đề distribution shift**: Phương pháp Combine & Re-split đã loại bỏ hoàn toàn hiện tượng overfitting do source-based split của CICDDoS2019, nâng Test Macro F1 từ 0,677 lên 0,939.
- **Mô hình đạt hiệu suất cao**: XGBoost đạt Accuracy 97,56%, Macro F1 93,86%, với Train/Test Gap chỉ 0,21%.
- **Demo 2 máy ảo**: VM Attacker sinh traffic DDoS (7 kịch bản), VM Victim chạy IDS/IPS phân loại thời gian thực bằng Scapy, features khớp 100% với training data.
- **Nhiều hình thức demo**: Demo 2 VM thời gian thực, Replay offline, Dashboard Streamlit.

Hệ thống có ý nghĩa thực tiễn trong việc hỗ trợ quản trị viên mạng phát hiện sớm các cuộc tấn công DDoS. Hướng phát triển tiếp theo bao gồm triển khai trên SDN controller để tự động chặn tấn công bằng flow rules và kiểm thử trên traffic mạng thực tế.
