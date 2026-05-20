# Phát hiện và ngăn chặn DDoS bằng Học máy

## Tổng quan

Đây là đề tài xây dựng hệ thống phát hiện, phân loại và hỗ trợ ngăn chặn tấn công DDoS bằng **Học máy** trên bộ dữ liệu **CICDDoS2019**. Dự án triển khai một quy trình nhiều lớp gồm: nạp dữ liệu, tiền xử lý, huấn luyện mô hình, đánh giá kết quả, giải thích mô hình bằng SHAP, giám sát lưu lượng thời gian thực bằng Scapy và bảng điều khiển trực quan bằng Streamlit.

**Tài liệu tham khảo**: Phương pháp được phát triển dựa trên bài thực hành Kaggle:
[DDoS Detection using Machine Learning](https://www.kaggle.com/code/rakibhossainsajib/ddos-detection-using-machine-learning)

## Tóm tắt công việc đã thực hiện

Trong đề tài **phát hiện và ngăn chặn tấn công DDoS bằng Học máy**, tôi đã thực hiện các công việc chính sau:

1. **Thu thập và chuẩn bị dữ liệu**
   - Sử dụng bộ dữ liệu **CICDDoS2019** ở định dạng Parquet.
   - Xây dựng module `data_loader.py` để tải, quét, lọc và gộp dữ liệu huấn luyện/kiểm thử.
   - Chỉ giữ lại các loại tấn công xuất hiện ở cả tập huấn luyện và tập kiểm thử.

2. **Tiền xử lý dữ liệu**
   - Chuẩn hóa tên nhãn giữa tập huấn luyện và tập kiểm thử.
   - Ánh xạ các nhãn thành 7 lớp: Benign, TCP SYN Flood, UDP Flood, UDP-Lag Flood, LDAP Flood, MSSQL Flood, NetBIOS Flood.
   - Xóa dữ liệu trùng lặp, xử lý giá trị thiếu, NaN và vô cực.
   - Loại bỏ các cột chỉ có một giá trị và các cột có tương quan cao.
   - Mã hóa nhãn bằng `LabelEncoder` và chuẩn hóa đặc trưng bằng `MinMaxScaler`.
   - Chia dữ liệu huấn luyện thành tập huấn luyện con và tập xác thực theo tỉ lệ 80/20.

3. **Xây dựng và huấn luyện mô hình**
   - Cài đặt và huấn luyện 5 mô hình: Random Forest, KNN, Extra Trees, MLP Classifier và XGBoost.
   - Lưu các mô hình đã huấn luyện vào thư mục `saved_models/`.
   - Toàn bộ quá trình được điều phối trong `main.py`.

4. **Đánh giá và so sánh kết quả**
   - Đánh giá mô hình bằng Accuracy, Precision, Recall và F1-score.
   - Tạo confusion matrix, ROC curve, classification report và biểu đồ so sánh mô hình.
   - Kết quả trên tập xác thực tốt nhất thuộc về **Random Forest** với F1-score khoảng **0.99345**.
   - Kết quả trên tập kiểm thử tốt nhất thuộc về **KNN** với F1-score khoảng **0.71172**.
   - Nhận xét được sự chênh lệch hiệu năng giữa tập xác thực và tập kiểm thử do phân phối dữ liệu kiểm thử khác với dữ liệu huấn luyện.

5. **Giải thích mô hình bằng SHAP**
   - Xây dựng `shap_explainer.py` để giải thích kết quả dự đoán.
   - Tạo SHAP summary plot, SHAP theo từng lớp tấn công và force plot cho dự đoán cụ thể.
   - Giúp phân tích những đặc trưng ảnh hưởng mạnh đến việc phân loại DDoS.

6. **Xây dựng hệ thống IPS thời gian thực**
   - Xây dựng `live_ips.py` để bắt gói tin bằng Scapy.
   - Gom các packet thành network flow, trích xuất đặc trưng và đưa vào mô hình ML để dự đoán.
   - Nếu phát hiện tấn công với độ tin cậy cao, hệ thống có thể gọi module mitigation để chặn IP.

7. **Xây dựng cơ chế giảm thiểu tấn công**
   - Xây dựng `mitigation.py` để chặn và bỏ chặn IP bằng firewall.
   - Hỗ trợ Linux với `iptables` và Windows với `netsh advfirewall`.
   - Có chế độ `SIMULATION_MODE` để demo an toàn, chỉ in lệnh mà không thực thi thật.

8. **Xây dựng bảng điều khiển giám sát**
   - Xây dựng giao diện `app.py` bằng Streamlit.
   - Hiển thị kết quả phân tích mô hình, bảng điểm, confusion matrix, ROC curve và SHAP.
   - Tạo màn hình Live Monitor để hiển thị log, cảnh báo tấn công và trạng thái block IP.
   - Tích hợp cấu hình cảnh báo Telegram.

9. **Đóng gói và triển khai**
   - Tạo `requirements.txt` để quản lý thư viện Python.
   - Tạo cấu hình deploy không dùng Docker gồm `Procfile`, `render.yaml`, `runtime.txt` và `.streamlit/config.toml`.

Tóm lại, đề tài đã hoàn thành một quy trình từ đầu đến cuối cho bài toán DDoS + Học máy: từ dữ liệu, tiền xử lý, huấn luyện, đánh giá, giải thích mô hình đến ứng dụng giám sát và phòng thủ thời gian thực.

## Các mô hình sử dụng

| STT | Mô hình | Mô tả | Tham số chính |
|-----|---------|-------|---------------|
| 1 | **Random Forest** | Tập hợp nhiều cây quyết định theo cơ chế bagging | `n_estimators=200`, `max_depth=25`, `min_samples_split=5` |
| 2 | **KNN** | Phân loại dựa trên các điểm láng giềng gần nhất | `n_neighbors=7`, `weights=distance`, `metric=minkowski` |
| 3 | **Extra Trees** | Mô hình cây cực ngẫu nhiên | `n_estimators=200`, `max_depth=25`, `min_samples_split=5` |
| 4 | **MLP Classifier** | Mạng nơ-ron nhiều lớp | `hidden_layer_sizes=(128,64)`, `solver=adam`, `early_stopping=True` |
| 5 | **XGBoost** | Mô hình cây tăng cường gradient | `n_estimators=200`, `max_depth=10`, `learning_rate=0.1`, `subsample=0.8` |

## Các loại lưu lượng và tấn công

| Lớp | Mô tả |
|-----|------|
| **TCP SYN Flood** | Tấn công tràn gói SYN |
| **UDP Flood** | Tấn công tràn lưu lượng UDP |
| **UDP-Lag Flood** | Tấn công UDP gây độ trễ |
| **LDAP Flood** | Tấn công khuếch đại qua LDAP |
| **MSSQL Flood** | Tấn công nhắm vào dịch vụ MSSQL |
| **NetBIOS Flood** | Tấn công liên quan đến NetBIOS |
| **Benign** | Lưu lượng bình thường, không phải tấn công |

## Kết quả so sánh mô hình

### Kết quả trên tập xác thực

| Mô hình | Accuracy | Precision | Recall | F1-score |
|---------|----------|-----------|--------|----------|
| Random Forest **[TỐT NHẤT]** | 0.993497 | 0.993461 | 0.993497 | 0.993451 |
| KNN | 0.992470 | 0.992395 | 0.992470 | 0.992412 |
| Extra Trees | 0.991786 | 0.992032 | 0.991786 | 0.991662 |
| MLP Classifier | 0.989604 | 0.989367 | 0.989604 | 0.989448 |
| XGBoost | 0.992256 | 0.992283 | 0.992256 | 0.992267 |

### Kết quả trên tập kiểm thử

| Mô hình | Accuracy | Precision | Recall | F1-score |
|---------|----------|-----------|--------|----------|
| Random Forest | 0.739710 | 0.813425 | 0.739710 | 0.645638 |
| KNN **[TỐT NHẤT]** | 0.745579 | 0.922581 | 0.745579 | 0.711721 |
| Extra Trees | 0.737823 | 0.812042 | 0.737823 | 0.643344 |
| MLP Classifier | 0.747285 | 0.687910 | 0.747285 | 0.709916 |
| XGBoost | 0.743666 | 0.843288 | 0.743666 | 0.695535 |

**Mô hình tốt nhất trên tập xác thực theo F1-score:** Random Forest  
**Mô hình tốt nhất trên tập kiểm thử theo F1-score:** KNN

> Lưu ý: Hiệu năng trên tập kiểm thử thấp hơn tập xác thực vì phân phối lớp của tập kiểm thử khác đáng kể so với tập huấn luyện. Đây là tình huống thực tế trong an ninh mạng, khi kiểu tấn công và tần suất tấn công có thể thay đổi theo thời gian.

## Kiến trúc hệ thống

```text
ML_DDOS/
|-- app.py                # Wrapper chạy frontend/app.py
|-- main.py               # Wrapper huấn luyện tương thích lệnh cũ
|-- model_audit.py        # Wrapper kiểm tra split/leakage/metrics
|-- requirements.txt      # Danh sách thư viện Python
|-- Procfile              # Start command cho nền tảng kiểu Heroku/Railway
|-- render.yaml           # Cấu hình deploy Render
|-- runtime.txt           # Phiên bản Python cho nền tảng deploy
|-- packages.txt          # System packages nếu nền tảng deploy hỗ trợ
|-- .streamlit/
|   `-- config.toml       # Cấu hình Streamlit server
|-- backend/
|   |-- main.py                # Entrypoint backend: huấn luyện
|   |-- model_audit.py         # Entrypoint backend: audit
|   |-- live_ips.py            # Entrypoint backend: IPS
|   |-- shap_explainer.py      # Entrypoint backend: SHAP
|   |-- requirements.txt       # Dependencies backend
|   `-- src/
|       `-- ml_ddos/
|       |-- data_loader.py      # Tải và nạp dữ liệu CICDDoS2019
|       |-- preprocessor.py     # Làm sạch, xử lý đặc trưng, mã hóa và chuẩn hóa dữ liệu
|       |-- models.py           # Định nghĩa, huấn luyện và đánh giá mô hình
|       |-- model_audit.py      # Kiểm tra per-class metrics, confusion matrix, CV
|       |-- shap_explainer.py   # Giải thích mô hình bằng SHAP
|       |-- live_ips.py         # Bộ máy IPS thời gian thực dựa trên Scapy
|       |-- mitigation.py       # Lệnh firewall để chặn/bỏ chặn IP
|       |-- paths.py            # Quản lý đường dẫn dùng chung khi deploy
|       `-- __init__.py
|-- frontend/
|   |-- app.py                 # Dashboard Streamlit
|   `-- requirements.txt       # Dependencies frontend
|-- data/                # Dữ liệu CICDDoS2019
|-- results/             # Biểu đồ, báo cáo và kết quả SHAP
|-- saved_models/        # Các mô hình đã huấn luyện
|-- audit_results/       # Báo cáo kiểm tra mô hình
|-- .gitignore
`-- README.md
```

Hệ thống được chia thành `backend/` và `frontend/` nhưng vẫn là modular monolith. Backend chứa toàn bộ logic dữ liệu, huấn luyện, audit, IPS và mitigation. Frontend chứa dashboard Streamlit. Các file Python ở root như `app.py`, `main.py`, `model_audit.py` là wrapper mỏng để giữ nguyên lệnh chạy cũ.

## Chức năng chính

### Giải thích mô hình bằng SHAP

- Tạo biểu đồ SHAP summary để xem đặc trưng nào ảnh hưởng mạnh đến kết quả phân loại.
- Tạo biểu đồ SHAP theo từng lớp tấn công.
- Tạo force plot để giải thích một dự đoán cụ thể.

### IPS thời gian thực (`live_ips.py`)

- Bắt gói tin trực tiếp bằng Scapy.
- Gom các packet thành network flow.
- Phân loại flow bằng mô hình ML đã huấn luyện.
- Tự động chặn IP tấn công khi độ tin cậy vượt ngưỡng cấu hình.
- Hỗ trợ chế độ mô phỏng `SIMULATION_MODE=True` để demo an toàn.

### Bảng điều khiển giám sát (`app.py`)

- Giao diện Streamlit nền tối.
- Tab phân tích để xem bảng điểm, biểu đồ so sánh, ROC curve, confusion matrix và SHAP.
- Tab giám sát trực tiếp để xem log, cảnh báo tấn công và trạng thái chặn IP.
- Tích hợp cấu hình gửi cảnh báo qua Telegram.

## Cài đặt

```bash
# Tạo môi trường ảo
python -m venv venv
source venv/bin/activate   # Linux/Mac
# venv\Scripts\activate    # Windows

# Cài đặt thư viện
pip install -r requirements.txt

# Cấu hình Kaggle API nếu cần tải dữ liệu từ Kaggle
# Đặt file kaggle.json vào thư mục ~/.kaggle/
```

## Cách sử dụng

```bash
# 1. Huấn luyện mô hình và sinh toàn bộ báo cáo
python main.py

# 2. Sinh biểu đồ giải thích SHAP
python shap_explainer.py

# 3. Chạy bảng điều khiển
streamlit run app.py

# 4. Chạy IPS ở chế độ mô phỏng an toàn
sudo python live_ips.py

# 5. Chạy IPS ở chế độ thật, có thể thực thi lệnh chặn IP
sudo python live_ips.py --live

# 6. Chỉ chạy bước nạp dữ liệu và tiền xử lý
python main.py --steps-1-2
```

## Triển khai không dùng Docker

Hệ thống deploy theo 2 phần:

- **Backend API**: FastAPI nhẹ tại `backend/api/index.py`, deploy lên Vercel.
- **Frontend dashboard**: Streamlit tại `frontend/app.py`, deploy lên Render.

Những tác vụ nặng như train model, audit full, SHAP, IPS sniffing và firewall mitigation nên chạy local hoặc trên worker/VPS riêng. Vercel backend chỉ dùng cho health check, đọc metrics, đọc audit summary và metadata.

### 1. Chuẩn bị trước khi deploy

Chạy local để bảo đảm đã có kết quả cho dashboard/API:

```bash
python backend/main.py
python backend/model_audit.py --models random_forest --cv-folds 3
```

Sau đó commit/push repo lên GitHub.

### 2. Deploy backend trên Vercel

Backend Vercel dùng các file:

```text
api/index.py
backend/api/index.py
backend/requirements-vercel.txt
vercel.json
.vercelignore
```

Các bước:

1. Vào Vercel, chọn **New Project**.
2. Import GitHub repo này.
3. Để **Root Directory** là root repo, không chọn riêng thư mục `backend`.
4. Vercel sẽ đọc `vercel.json` và dùng `api/index.py` làm Python Function entrypoint.
5. Thêm Environment Variable nếu cần:
   ```text
   ALLOWED_ORIGINS=*
   ```
   Khi có URL Render frontend, có thể đổi thành:
   ```text
   ALLOWED_ORIGINS=https://your-frontend.onrender.com
   ```
6. Deploy.
7. Test backend:
   ```text
   https://your-backend.vercel.app/health
   https://your-backend.vercel.app/docs
   ```

Các endpoint chính:

```text
GET /health
GET /model-info
GET /metrics/validation
GET /metrics/test
GET /audit/summary
GET /audit/weighted-metrics
GET /audit/per-class
GET /audit/class-distribution
GET /audit/leakage-checks
GET /manifest
```

### 3. Deploy frontend trên Render

Frontend Render dùng các file:

```text
frontend/app.py
requirements.txt
render.yaml
Procfile
.streamlit/config.toml
packages.txt
```

Các bước:

1. Vào Render, chọn **New Web Service**.
2. Chọn repo này.
3. Render có thể tự đọc `render.yaml`. Nếu cấu hình thủ công:
   - Build Command:
     ```bash
     pip install -r requirements.txt
     ```
   - Start Command:
     ```bash
     streamlit run frontend/app.py --server.address=0.0.0.0 --server.port=$PORT --server.headless=true
     ```
4. Thêm Environment Variable:
   ```text
   BACKEND_API_URL=https://your-backend.vercel.app
   ```
5. Deploy.
6. Mở dashboard Render URL, sidebar sẽ hiển thị trạng thái Backend API.

### Deploy trên Railway/Heroku-like

Các nền tảng hỗ trợ `Procfile` có thể dùng command:

```text
web: streamlit run frontend/app.py --server.address=0.0.0.0 --server.port=$PORT --server.headless=true
```

### Deploy trên Streamlit Community Cloud

1. Đẩy project lên GitHub.
2. Chọn app file:
   ```text
   frontend/app.py
   ```
3. Streamlit sẽ cài dependencies từ `requirements.txt`.
4. Nếu cần system packages, nền tảng sẽ đọc `packages.txt`.

### Chạy local giống môi trường deploy

```bash
set BACKEND_API_URL=https://your-backend.vercel.app
streamlit run frontend/app.py --server.address=0.0.0.0 --server.port=8501 --server.headless=true
```

Git Bash/Linux/Mac:

```bash
export BACKEND_API_URL=https://your-backend.vercel.app
streamlit run frontend/app.py --server.address=0.0.0.0 --server.port=8501 --server.headless=true
```

Mở:

```text
http://localhost:8501
```

## Báo cáo và biểu đồ đã sinh

Tất cả kết quả được lưu trong thư mục `results/`.

| File | Ý nghĩa |
|------|--------|
| `*_confusion_matrix.png` | Ma trận nhầm lẫn của từng mô hình |
| `*_roc_curve.png` | Đường ROC theo từng lớp |
| `*_feature_importance.png` | Top đặc trưng quan trọng của các mô hình cây |
| `*_shap_summary_bar.png` | Độ quan trọng đặc trưng theo SHAP |
| `*_shap_<class>.png` | SHAP beeswarm theo từng lớp tấn công |
| `*_shap_force_plot.png` | Force plot cho một dự đoán cụ thể |
| `model_comparison.png` | Biểu đồ so sánh các mô hình |
| `*_classification_report.txt` | Báo cáo precision, recall và F1 theo từng lớp |
| `validation_scores.csv` / `test_scores.csv` | Bảng điểm đánh giá ở định dạng CSV |

Các mô hình đã huấn luyện được lưu trong thư mục `saved_models/` dưới dạng `.pkl`.

## Bộ dữ liệu

**CICDDoS2019** do Canadian Institute for Cybersecurity công bố.

- Nguồn dữ liệu: https://www.kaggle.com/datasets/dhoogla/cicddos2019
- Định dạng: file Parquet, chia theo tập huấn luyện và kiểm thử cho từng loại tấn công.
- Số đặc trưng ban đầu: 78 đặc trưng lưu lượng mạng.
- Sau tiền xử lý: còn 32 đặc trưng, sau khi loại bỏ 12 cột chỉ có một giá trị và 33 cột tương quan cao.

## Quy trình tiền xử lý

1. **Chuẩn hóa nhãn**: đồng bộ tên nhãn giữa tập huấn luyện và tập kiểm thử.
2. **Xóa dữ liệu trùng lặp**: loại bỏ các dòng bị lặp.
3. **Xử lý giá trị không hợp lệ**: thay `inf` và `NaN` bằng giá trị trung vị của cột.
4. **Loại bỏ cột một giá trị**: xóa các đặc trưng không mang thông tin phân loại.
5. **Loại bỏ cột tương quan cao**: xóa các cột có tương quan lớn hơn 0.8.
6. **Mã hóa nhãn**: dùng `LabelEncoder` cho 7 lớp đầu ra.
7. **Chuẩn hóa đặc trưng**: dùng `MinMaxScaler` được khớp trên tập huấn luyện.
8. **Chia huấn luyện/xác thực**: tách 80/20 có giữ tỉ lệ lớp.

## Thước đo đánh giá

Mỗi mô hình được đánh giá bằng:

- Accuracy, Precision, Recall và F1-score theo trung bình có trọng số.
- Ma trận nhầm lẫn.
- Đường ROC theo từng lớp.
- Biểu đồ độ quan trọng đặc trưng cho các mô hình dạng cây.
- Biểu đồ giải thích SHAP.
