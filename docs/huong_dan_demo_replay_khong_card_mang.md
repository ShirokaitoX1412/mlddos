# Hướng Dẫn Demo IDS/IPS Không Cần Card Mạng

Tài liệu này mô tả kịch bản demo phù hợp khi không thể dùng card mạng, không dùng dual boot và không dùng máy ảo. Thay vì bắt gói tin trực tiếp bằng Npcap/Scapy, hệ thống phát lại các bản ghi flow từ tập kiểm thử CICDDoS2019 như một luồng sự kiện thời gian thực.

## 1. Mục Tiêu Demo

Mục tiêu của kịch bản này là chứng minh mô hình học máy đã được tích hợp vào một pipeline IDS/IPS vận hành được:

```text
CICDDoS2019 testing flows
        |
        v
Replay từng flow theo thời gian
        |
        v
selected_model.pkl dự đoán
        |
        v
Ghi results/live_events.csv
        |
        v
Streamlit Dashboard hiển thị cảnh báo
```

Kịch bản này không chứng minh khả năng bắt gói tin thật từ card mạng. Thay vào đó, nó kiểm chứng khả năng đưa flow vào mô hình, phân loại, ghi log và hiển thị cảnh báo.

## 2. Khi Nào Nên Dùng

Nên dùng replay demo khi:

- Card mạng vật lý bị lỗi hoặc không ổn định.
- Không muốn dùng dual boot.
- Không muốn dùng máy ảo.
- Kh?ng mu?n ph? thu?c Npcap ho?c card m?ng v?t l?.
- Cần một demo dễ chạy, có thể trình bày ổn định trong bảo vệ đồ án.

## 3. Chuẩn Bị

Kích hoạt môi trường Python:

```powershell
cd D:\source\CyRadar\mlddos
.\venv\Scripts\activate
```

Nếu chưa cài thư viện:

```powershell
pip install -r requirements.txt
```

Kiểm tra model đã tồn tại:

```powershell
dir saved_models\selected_model.pkl
```

Nếu chưa có model:

```powershell
python main.py
```

## 4. Chạy Replay Demo

Chạy replay 500 flow với tốc độ 0.2 giây mỗi flow:

```powershell
python replay_ips.py --speed 0.2 --limit 500
```

Chạy nhanh 20 flow để kiểm thử:

```powershell
python replay_ips.py --limit 20 --speed 0
```

Chạy và xáo trộn thứ tự flow:

```powershell
python replay_ips.py --limit 500 --speed 0.2 --shuffle
```

Chạy từ một file parquet/csv cụ thể:

```powershell
python replay_ips.py --source data\Syn-testing.parquet --limit 200 --speed 0.1
```

Mặc định, mỗi lần chạy replay sẽ ghi mới file:

```text
results/live_events.csv
```

Nếu muốn ghi nối tiếp log cũ:

```powershell
python replay_ips.py --limit 100 --append
```

## 5. Chạy Dashboard

Mở terminal khác:

```powershell
.\venv\Scripts\activate
streamlit run app.py
```

Vào tab:

```text
Live Monitor
```

Dashboard sẽ đọc:

```text
results/live_events.csv
```

và hiển thị các event được replay như cảnh báo thời gian thực.

## 6. Kiểm Tra Kết Quả

Kiểm tra file event:

```powershell
Get-Content results\live_events.csv -Tail 10
```

Kiểm tra cú pháp:

```powershell
python -m py_compile backend\src\ml_ddos\replay_ips.py frontend\app.py
```

Kiểm tra CLI:

```powershell
python replay_ips.py --help
```

## 7. Cách Viết Vào Báo Cáo

Tên mục đề xuất:

```text
Kịch bản mô phỏng giám sát thời gian thực dựa trên replay flow
```

Đoạn mô tả có thể dùng:

> Do giới hạn phần cứng, hệ thống demo được triển khai theo cơ chế replay flow. Các bản ghi flow từ tập kiểm thử CICDDoS2019 được phát lại tuần tự như luồng dữ liệu thời gian thực, sau đó đưa qua mô hình học máy đã huấn luyện để phân loại và ghi log cảnh báo. Cách tiếp cận này cho phép kiểm chứng pipeline vận hành của hệ thống IDS/IPS mà không phụ thuộc vào card mạng vật lý.

Phần giới hạn nên nêu rõ:

> Kịch bản replay không thay thế hoàn toàn việc bắt gói tin trong môi trường mạng thật. Tuy nhiên, nó phù hợp để kiểm chứng luồng xử lý của hệ thống từ dữ liệu flow đầu vào, mô hình dự đoán, ghi nhận cảnh báo đến hiển thị dashboard.

## 8. So Sánh Với Live Capture

| Tiêu chí | Replay demo | Live capture |
|---|---|---|
| Cần card mạng | Không | Có |
| Cần Npcap | Không | Có |
| Cần máy ảo/dual boot | Không | Không bắt buộc |
| Độ ổn định khi bảo vệ | Cao | Phụ thuộc môi trường |
| Chứng minh bắt packet thật | Không | Có |
| Chứng minh pipeline IDS/IPS | Có | Có |

