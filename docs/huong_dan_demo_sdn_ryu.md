# Hướng dẫn demo SDN/Ryu cho hệ thống phát hiện DDoS

## 1. Mục tiêu demo

Kịch bản này triển khai hệ thống IDS/IPS trên mạng SDN mô phỏng bằng Mininet:

```text
Host benign / Host attacker / Host victim
        ↓
Open vSwitch trong Mininet
        ↓
Ryu Controller
        ↓
Thu thập OpenFlow flow statistics
        ↓
Trích xuất đặc trưng CICDDoS-like
        ↓
selected_model.pkl dự đoán Benign / DDoS
        ↓
Ghi results/live_events.csv
        ↓
Tuỳ chọn cài OpenFlow drop rule để chặn luồng tấn công
```

Demo này phù hợp để trình bày cơ chế đưa mô hình học máy vào tầng điều khiển SDN. Mặc định controller chạy ở chế độ quan sát, chưa chặn thật, để tránh làm gián đoạn mạng khi mới kiểm thử.

## 2. Thành phần đã bổ sung

- `sdn_ryu_detector.py`: entrypoint ở thư mục gốc để chạy bằng `ryu-manager`.
- `backend/src/ml_ddos/sdn_ryu_detector.py`: Ryu controller chính.
- `tools/mininet_sdn_topology.py`: topology Mininet gồm 1 switch, 1 client benign, 1 victim, 1 attacker.
- `requirements-sdn.txt`: dependency riêng cho Ryu.
- `results/live_events.csv`: log dùng chung với dashboard Streamlit.

## 3. Chuẩn bị Ubuntu VM

Khuyến nghị dùng Ubuntu 22.04 LTS hoặc môi trường có Python 3.10. Ryu 4.34 khá cũ, dễ lỗi trên Python 3.14 và setuptools mới.

Cách nhanh bằng script:

```bash
cd /media/sf_mlddos
chmod +x tools/setup_sdn_ubuntu.sh tools/run_sdn_controller.sh
./tools/setup_sdn_ubuntu.sh
```

Cách thủ công:

Cài gói hệ thống:

```bash
sudo apt update
sudo apt install -y mininet openvswitch-switch hping3 iperf3 curl tcpdump \
  python3 python3-venv python3-dev build-essential libffi-dev libssl-dev
```

Kiểm tra dịch vụ Open vSwitch:

```bash
sudo systemctl enable --now openvswitch-switch
sudo ovs-vsctl show
```

Nếu Ubuntu của bạn chỉ có Python 3.14 và cài Ryu bị lỗi, nên dùng Ubuntu 22.04 hoặc cài Python 3.10 bằng pyenv. Không nên cố chạy Ryu 4.34 trực tiếp trên Python 3.14.

## 4. Cài môi trường Python

Vào thư mục project trong Ubuntu VM. Nếu dùng shared folder VirtualBox, ví dụ:

```bash
cd /media/sf_mlddos
```

Tạo môi trường ảo:

```bash
python3 -m venv ryu-venv
source ryu-venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-sdn.txt
```

Nếu Ryu lỗi do setuptools, chạy lại:

```bash
pip install "setuptools==57.5.0"
pip install --no-build-isolation ryu==4.34
```

## 5. Chuẩn bị model

Kiểm tra model đã train:

```bash
ls saved_models/selected_model.pkl
```

Nếu chưa có, train lại:

```bash
python main.py
```

## 6. Chạy Ryu controller

Chạy chế độ quan sát, chưa chặn:

```bash
source ryu-venv/bin/activate
export MLDDOS_OBSERVE_ONLY=1
export MLDDOS_THRESHOLD=0.95
export MLDDOS_POLL_INTERVAL=3
export MLDDOS_PPS_THRESHOLD=800
ryu-manager sdn_ryu_detector.py
```

Hoặc dùng script:

```bash
source ryu-venv/bin/activate
./tools/run_sdn_controller.sh
```

Nếu muốn bắt đầu một phiên demo với log sạch:

```bash
MLDDOS_RESET_EVENTS=1 ./tools/run_sdn_controller.sh
```

Ý nghĩa cấu hình:

- `MLDDOS_OBSERVE_ONLY=1`: chỉ ghi cảnh báo, không chặn.
- `MLDDOS_OBSERVE_ONLY=0`: cài OpenFlow drop rule khi phát hiện tấn công.
- `MLDDOS_THRESHOLD`: ngưỡng confidence để chặn.
- `MLDDOS_POLL_INTERVAL`: chu kỳ đọc flow statistics.
- `MLDDOS_PPS_THRESHOLD`: ngưỡng packet/second bổ trợ cho SDN vì OpenFlow statistics không chứa đủ TCP flags như dữ liệu CICDDoS.

## 7. Chạy topology Mininet

Mở terminal Ubuntu thứ hai:

```bash
cd /media/sf_mlddos
sudo python3 tools/mininet_sdn_topology.py
```

Trong Mininet CLI, chạy server ở victim:

```bash
h2 python3 -m http.server 80 &
```

Tạo traffic bình thường:

```bash
h1 curl http://10.0.0.2
h1 ping -c 3 10.0.0.2
```

Tạo traffic mô phỏng SYN flood:

```bash
h3 hping3 -S --flood -p 80 10.0.0.2
```

Hoặc UDP flood:

```bash
h3 hping3 --udp --flood -p 80 10.0.0.2
```

Sau vài giây, controller sẽ ghi sự kiện vào:

```bash
tail -f results/live_events.csv
```

Nếu file còn log từ phiên cũ, dừng Ryu rồi chạy lại controller với `MLDDOS_RESET_EVENTS=1`.

## 8. Bật chặn thật bằng OpenFlow

Sau khi observe-only hoạt động ổn, dừng Ryu và chạy lại:

```bash
export MLDDOS_OBSERVE_ONLY=0
export MLDDOS_THRESHOLD=0.95
ryu-manager sdn_ryu_detector.py
```

Nếu dùng script:

```bash
MLDDOS_OBSERVE_ONLY=0 MLDDOS_THRESHOLD=0.95 ./tools/run_sdn_controller.sh
```

Khi flow bị đánh dấu tấn công và confidence vượt ngưỡng, controller sẽ cài rule drop priority cao trên Open vSwitch. Kiểm tra rule:

```bash
sudo ovs-ofctl -O OpenFlow13 dump-flows s1
```

Xoá mạng Mininet sau khi test:

```bash
sudo mn -c
```

## 9. Chạy dashboard

Có hai lựa chọn giao diện:

- Dashboard tổng của project:

```bash
streamlit run frontend/app.py
```

Vào tab Live Monitor để xem cảnh báo sinh ra từ Ryu controller.

- Dashboard riêng cho demo SDN:

```bash
streamlit run frontend/sdn_dashboard.py
```

Dashboard SDN hiển thị sơ đồ topology, số lượng cảnh báo, số flow bị chặn, phân bố loại tấn công, timeline confidence và bảng live events. Ryu và Mininet vẫn cần chạy trong terminal vì đây là các tiến trình điều khiển network stack Linux, còn dashboard là lớp giao diện để trình bày kết quả mô phỏng.

## 10. Cách viết vào báo cáo

Tên kịch bản nên dùng:

```text
Kịch bản phát hiện và giảm thiểu DDoS trong mạng SDN sử dụng Ryu Controller
```

Mô tả ngắn:

```text
Hệ thống SDN được mô phỏng bằng Mininet và Open vSwitch. Ryu Controller đóng vai trò bộ điều khiển trung tâm, vừa thực hiện chức năng learning switch, vừa thu thập thống kê luồng qua OpenFlow. Các thống kê này được chuyển đổi thành tập đặc trưng gần tương ứng với dữ liệu CICDDoS2019 và đưa vào mô hình học máy đã huấn luyện. Khi phát hiện luồng có khả năng là DDoS, hệ thống ghi cảnh báo và có thể cài đặt luật OpenFlow có độ ưu tiên cao để chặn luồng tấn công.
```

Lưu ý kỹ thuật cần viết rõ:

```text
OpenFlow flow statistics không chứa đầy đủ các trường chi tiết như TCP flags, inter-arrival time theo từng gói hoặc phân bố kích thước gói giống dữ liệu CICDDoS2019. Vì vậy, demo SDN sử dụng các đặc trưng có thể suy ra từ thống kê luồng và bổ sung ngưỡng packet-rate để tăng độ ổn định khi phát hiện flood trong môi trường Mininet.
```

## 11. Kiểm thử nhanh

```bash
python -m py_compile backend/src/ml_ddos/sdn_ryu_detector.py sdn_ryu_detector.py
python sdn_ryu_detector.py
sudo python3 tools/mininet_sdn_topology.py
```

Nếu lệnh `python sdn_ryu_detector.py` báo Ryu chưa cài thì bình thường trên máy Windows. Trên Ubuntu VM, controller phải chạy bằng `ryu-manager sdn_ryu_detector.py`.
