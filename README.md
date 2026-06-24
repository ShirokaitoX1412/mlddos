# DDoS IDS/IPS sử dụng học máy

Dự án xây dựng hệ thống phát hiện và hỗ trợ ngăn chặn tấn công DDoS bằng mô hình học máy. Phiên bản hiện tại tập trung vào luồng chính: huấn luyện mô hình trên CICDDoS2019, chạy IDS/IPS trên máy nạn nhân, sinh lưu lượng kiểm thử từ máy attacker và quan sát log trên dashboard Streamlit.

## 1. Thành phần chính

```text
mlddos/
├── app.py                         # Launcher dashboard / cài app nạn nhân
├── live_ips.py                    # Entrypoint IDS/IPS live
├── main.py                        # Entrypoint train model
├── backend/src/ml_ddos/
│   ├── data_loader.py             # Nạp dữ liệu CICDDoS2019
│   ├── preprocessor.py            # Làm sạch và tiền xử lý dữ liệu
│   ├── models.py                  # Huấn luyện và đánh giá mô hình
│   ├── main.py                    # Pipeline train chính
│   ├── live_ips.py                # Agent IDS/IPS bắt gói, trích đặc trưng, dự đoán
│   ├── mitigation.py              # Chặn IP bằng iptables/nftables/Windows Firewall
│   ├── alert_notifier.py          # Thông báo cảnh báo cục bộ
│   └── paths.py                   # Đường dẫn dùng chung
├── frontend/app.py                # Dashboard Streamlit
├── tools/ddos_traffic_generator.py # Sinh lưu lượng kiểm thử trên máy attacker
├── data/                          # Dữ liệu CICDDoS2019
├── results/                       # Log live và kết quả chạy hiện tại
├── saved_models/                  # Mô hình đã huấn luyện
├── tests/                         # Smoke test
└── requirements.txt
```

## 2. Cài đặt

```bash
python -m venv venv
source venv/bin/activate      # Linux/Kali/Ubuntu
# hoặc
venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

Trên máy nạn nhân Linux/Kali:

```bash
sudo apt update
sudo apt install -y tcpdump iptables
pip install cicflowmeter
```

## 3. Huấn luyện mô hình

```bash
python main.py --combine-resplit --cv-folds 3 --search-iter 10
```

Mô hình triển khai chính:

```text
saved_models/selected_model.pkl
```

## 4. Chạy dashboard

```bash
python app.py
```

Hoặc:

```bash
streamlit run frontend/app.py
```

Dashboard chỉ tập trung vào demo live:

- Tổng số sự kiện IDS/IPS.
- Số tấn công phát hiện.
- Số IP đã chặn.
- Trạng thái IPS.
- Nhật ký phát hiện theo thời gian thực.

## 5. Demo hai máy ảo

Ví dụ IP:

- Attacker thật: `192.168.56.101`
- Máy nạn nhân: `192.168.56.103`
- IP attacker giả lập: `10.10.1.10` đến `10.10.1.14`

Chạy IDS/IPS trên máy nạn nhân ở chế độ quan sát:

```bash
sudo $(which python) live_ips.py \
  --interface eth1 \
  --victim-ip 192.168.56.103 \
  --model selected_model \
  --threshold 0.80 \
  --cicflowmeter \
  --fast-log \
  --clear-events \
  --cicflowmeter-cmd 'cicflowmeter -f {pcap} -c {csv}'
```

Chạy IPS chặn thật:

```bash
sudo $(which python) live_ips.py \
  --interface eth1 \
  --victim-ip 192.168.56.103 \
  --model selected_model \
  --threshold 0.80 \
  --cicflowmeter \
  --fast-log \
  --live \
  --cicflowmeter-cmd 'cicflowmeter -f {pcap} -c {csv}'
```

Kiểm tra luật chặn:

```bash
sudo iptables -L INPUT -n --line-numbers
```

## 6. Sinh lưu lượng kiểm thử trên máy attacker

TCP SYN Flood:

```bash
sudo $(which python) tools/ddos_traffic_generator.py \
  --target 192.168.56.103 \
  --attack syn_flood \
  --duration 30 \
  --pps 300
```

Tấn công hỗn hợp nhiều IP giả lập:

```bash
sudo $(which python) tools/ddos_traffic_generator.py \
  --target 192.168.56.103 \
  --attack mixed \
  --duration 60 \
  --pps 600 \
  --spoof-sources 10.10.1.10,10.10.1.11,10.10.1.12,10.10.1.13,10.10.1.14
```

## 7. Kiểm tra nhanh

```bash
python -m py_compile backend/src/ml_ddos/live_ips.py frontend/app.py tools/ddos_traffic_generator.py
python live_ips.py --help
python main.py --help
```

## 8. Ghi chú

Demo chỉ dùng trong môi trường lab có kiểm soát. Các lệnh sinh traffic phục vụ kiểm thử hệ thống IDS/IPS, không dùng trên hệ thống không được phép.
