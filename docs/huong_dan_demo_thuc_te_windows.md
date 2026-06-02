# Hướng Dẫn Demo Thực Tế DDoS IDS/IPS Trên 1 Máy Windows

Tài liệu này mô tả cách triển khai demo hệ thống phát hiện và hỗ trợ ngăn chặn DDoS trên một máy Windows, không sử dụng máy ảo. Mục tiêu là chứng minh luồng xử lý thực tế: bắt gói tin, gom thành flow, trích xuất đặc trưng, dự đoán bằng mô hình học máy, ghi log cảnh báo và tùy chọn chặn IP bằng Windows Firewall.

## 1. Kiến Trúc Demo

```text
Traffic trên máy Windows
        |
        v
Npcap + Scapy
        |
        v
Flow Aggregator
        |
        v
Trích xuất đặc trưng CICDDoS-like
        |
        v
ML Pipeline model
        |
        v
Dự đoán Benign / DDoS
        |
        v
Log CSV + tùy chọn chặn IP bằng Windows Firewall
        |
        v
Streamlit Dashboard
```

Thành phần chính:

- `live_ips.py`: bắt gói tin thời gian thực, gom packet thành flow, dự đoán và ghi log.
- `saved_models/selected_model.pkl`: mô hình được chọn sau quá trình huấn luyện.
- `results/live_events.csv`: file log sự kiện thật để dashboard đọc.
- `frontend/app.py`: dashboard Streamlit hiển thị kết quả huấn luyện, audit và live events.
- `backend/src/ml_ddos/mitigation.py`: gọi Windows Firewall bằng `netsh advfirewall` khi bật chế độ live.

## 2. Chuẩn Bị Môi Trường

### 2.1. Cài Python dependencies

Nên dùng virtual environment của project:

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

Nếu dependencies backend/frontend tách riêng, có thể cài thêm:

```powershell
pip install -r backend\requirements.txt
pip install -r frontend\requirements.txt
```

### 2.2. Cài Npcap

Cần cài Npcap trên Windows để Scapy bắt được packet.

Khuyến nghị khi cài:

- Chọn chế độ WinPcap API-compatible nếu trình cài có tùy chọn này.
- Bật hỗ trợ loopback adapter nếu có tùy chọn.
- Khởi động lại terminal sau khi cài.

### 2.3. Kiểm tra model

Kiểm tra đã có model được train:

```powershell
dir saved_models
```

Nếu chưa có `selected_model.pkl`, chạy:

```powershell
python main.py
```

## 3. Chạy Live IPS Ở Chế Độ An Toàn

### 3.1. Liệt kê interface

```powershell
python live_ips.py --list-interfaces
```

Nếu demo trên chính một máy Windows, thử interface:

- `Npcap Loopback Adapter`
- Hoặc card Wi-Fi/Ethernet đang kết nối mạng

### 3.2. Chạy IPS simulation mode

Simulation mode là chế độ mặc định. Hệ thống chỉ ghi log và in lệnh chặn IP, không thực thi firewall.

```powershell
python live_ips.py --interface "Npcap Loopback Adapter" --threshold 0.95
```

Nếu dùng card mạng thật:

```powershell
python live_ips.py --interface "<interface-name>" --threshold 0.95
```

Sau khi có flow được phân tích, log sẽ được ghi vào:

```text
results/live_events.csv
```

## 4. Tạo Traffic Demo Trên 1 Máy

### 4.1. Tạo web server nội bộ

Mở terminal thứ nhất:

```powershell
python -m http.server 8080
```

Mở trình duyệt vào:

```text
http://127.0.0.1:8080
```

### 4.2. Tạo request lặp để có traffic

Mở terminal thứ hai:

```powershell
for ($i=0; $i -lt 200; $i++) { Invoke-WebRequest http://127.0.0.1:8080 -UseBasicParsing | Out-Null }
```

Hoặc dùng ping để tạo traffic ICMP:

```powershell
ping 127.0.0.1 -n 50
```

Lưu ý: dataset gốc tập trung vào flow TCP/UDP, nên HTTP request sẽ phù hợp hơn ICMP cho demo.

## 5. Chạy Dashboard

Mở terminal mới:

```powershell
.\venv\Scripts\activate
streamlit run app.py
```

Trong tab Live Monitor:

- Nếu `results/live_events.csv` đã có dữ liệu, dashboard sẽ hiển thị live events thật.
- Nếu chưa có dữ liệu, dashboard vẫn cho phép sinh demo events bằng nút `Start IPS (Demo)`.

## 6. Demo Chặn IP Thật Bằng Windows Firewall

Chỉ dùng bước này sau khi simulation mode đã chạy ổn định.

Mở PowerShell bằng quyền Administrator, sau đó chạy:

```powershell
python live_ips.py --interface "<interface-name>" --threshold 0.99 --live
```

Tham số `--live` sẽ cho phép module mitigation thực thi lệnh:

```powershell
netsh advfirewall firewall add rule name="BLOCK_DDoS_<ip>" dir=in action=block remoteip=<ip> protocol=any enable=yes
```

Để xóa rule chặn IP:

```powershell
netsh advfirewall firewall delete rule name="BLOCK_DDoS_<ip>"
```

Khuyến nghị khi báo cáo/đồ án:

- Demo chính nên dùng simulation mode để an toàn.
- Chỉ bật `--live` khi cần chứng minh khả năng tích hợp firewall.
- Dùng threshold cao như `0.99` để giảm nguy cơ chặn nhầm.

## 7. Kiểm Thử Nhanh

Kiểm tra cú pháp:

```powershell
python -m py_compile backend\src\ml_ddos\live_ips.py frontend\app.py
```

Kiểm tra interface:

```powershell
python live_ips.py --list-interfaces
```

Kiểm tra dashboard:

```powershell
streamlit run app.py
```

Kiểm tra file log:

```powershell
Get-Content results\live_events.csv -Tail 10
```

## 8. Ghi Chú Về WSL

WSL có thể dùng để train model, xử lý dữ liệu hoặc chạy các script offline. Tuy nhiên, với demo thời gian thực trên Windows, không nên đặt thành phần live capture/firewall trong WSL vì:

- WSL không nằm trực tiếp trên toàn bộ network stack của Windows.
- Việc bắt gói trên interface Windows có thể không đầy đủ hoặc không ổn định.
- Windows Firewall nên được gọi trực tiếp từ Windows native PowerShell.

Vì vậy, cấu hình khuyến nghị là:

- Train/offline: Windows hoặc WSL đều được.
- Live IPS: Windows native.
- Firewall: Windows native, Administrator PowerShell.
- Dashboard: Windows native để đọc cùng `results/live_events.csv`.

