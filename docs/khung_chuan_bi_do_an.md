# Khung Chuẩn Bị Đồ Án Tốt Nghiệp

**Đề tài:** Nghiên cứu xây dựng giải pháp phát hiện DDoS sử dụng học máy

## 1. Bảng Thuật Ngữ Và Chữ Viết Tắt

| STT | Thuật ngữ / Chữ viết tắt | Tiếng Anh / Dạng đầy đủ | Giải thích |
|---:|---|---|---|
| 1 | ATTT | An toàn thông tin | Lĩnh vực nghiên cứu các biện pháp bảo vệ hệ thống, dữ liệu và dịch vụ trước các nguy cơ mất an toàn. |
| 2 | DoS | Denial of Service | Tấn công từ chối dịch vụ nhằm làm gián đoạn khả năng phục vụ của hệ thống mục tiêu. |
| 3 | DDoS | Distributed Denial of Service | Tấn công từ chối dịch vụ phân tán, sử dụng nhiều nguồn phát sinh lưu lượng để làm quá tải hệ thống mục tiêu. |
| 4 | Botnet | Bot Network | Mạng lưới các thiết bị bị chiếm quyền điều khiển, thường được dùng để phát động tấn công DDoS. |
| 5 | SYN Flood | TCP SYN Flood | Kiểu tấn công gửi số lượng lớn gói SYN nhằm làm cạn kiệt tài nguyên xử lý kết nối TCP. |
| 6 | UDP Flood | User Datagram Protocol Flood | Kiểu tấn công gửi lượng lớn gói UDP đến hệ thống mục tiêu nhằm tiêu hao băng thông hoặc tài nguyên xử lý. |
| 7 | LDAP Flood | Lightweight Directory Access Protocol Flood | Dạng tấn công tạo lưu lượng lớn liên quan đến dịch vụ LDAP. |
| 8 | MSSQL Flood | Microsoft SQL Server Flood | Dạng tấn công tạo lưu lượng bất thường nhắm vào hoặc lợi dụng dịch vụ Microsoft SQL Server. |
| 9 | NetBIOS Flood | Network Basic Input/Output System Flood | Dạng tấn công liên quan đến giao thức hoặc dịch vụ NetBIOS. |
| 10 | Flow | Network Flow | Luồng mạng được mô tả bởi tập thông tin như địa chỉ IP, cổng, giao thức, số packet, số byte và thời lượng. |
| 11 | Packet | Network Packet | Đơn vị dữ liệu được truyền qua mạng, gồm phần header và có thể có payload. |
| 12 | PCAP | Packet Capture | Định dạng hoặc quá trình thu thập gói tin mạng phục vụ phân tích lưu lượng. |
| 13 | Feature | Đặc trưng | Biến đầu vào dùng để mô tả một mẫu dữ liệu, ví dụ số lượng gói tin, tốc độ truyền hoặc kích thước gói tin. |
| 14 | Dataset | Bộ dữ liệu | Tập hợp dữ liệu được sử dụng để huấn luyện, kiểm thử và đánh giá mô hình học máy. |
| 15 | CICDDoS2019 | Canadian Institute for Cybersecurity DDoS 2019 | Bộ dữ liệu phục vụ nghiên cứu phát hiện tấn công DDoS. |
| 16 | ML | Machine Learning | Học máy, lĩnh vực cho phép mô hình học quy luật từ dữ liệu để đưa ra dự đoán. |
| 17 | Supervised Learning | Học có giám sát | Phương pháp học máy sử dụng dữ liệu có nhãn để huấn luyện mô hình. |
| 18 | Classification | Phân loại | Bài toán dự đoán nhãn lớp của một mẫu dữ liệu đầu vào. |
| 19 | RF | Random Forest | Mô hình tập hợp nhiều cây quyết định, thường dùng cho phân loại dữ liệu dạng bảng. |
| 20 | ET | Extra Trees | Mô hình cây cực kỳ ngẫu nhiên, tăng tính đa dạng giữa các cây trong ensemble. |
| 21 | KNN | K-Nearest Neighbors | Thuật toán phân loại dựa trên nhãn của các điểm dữ liệu lân cận gần nhất. |
| 22 | MLP | Multi-Layer Perceptron | Mạng nơ-ron truyền thẳng nhiều lớp, có khả năng học quan hệ phi tuyến. |
| 23 | XGBoost | Extreme Gradient Boosting | Thuật toán boosting hiệu quả cao cho các bài toán phân loại dữ liệu bảng. |
| 24 | Pipeline | Quy trình xử lý | Chuỗi các bước tiền xử lý, huấn luyện và đánh giá được tổ chức có thứ tự. |
| 25 | Data Leakage | Rò rỉ dữ liệu | Hiện tượng thông tin từ tập kiểm thử hoặc validation bị sử dụng trong quá trình huấn luyện. |
| 26 | Cross-validation | Kiểm định chéo | Kỹ thuật chia dữ liệu thành nhiều fold để đánh giá độ ổn định của mô hình. |
| 27 | SMOTE | Synthetic Minority Over-sampling Technique | Kỹ thuật sinh mẫu tổng hợp cho lớp thiểu số nhằm giảm mất cân bằng dữ liệu. |
| 28 | Accuracy | Độ chính xác tổng thể | Tỷ lệ mẫu được dự đoán đúng trên tổng số mẫu. |
| 29 | Precision | Độ chính xác dự đoán dương | Tỷ lệ dự đoán dương đúng trên tổng số mẫu được dự đoán là dương. |
| 30 | Recall | Độ bao phủ / Độ nhạy | Tỷ lệ mẫu dương thực tế được mô hình phát hiện đúng. |
| 31 | F1-Score | F1 | Trung bình điều hòa giữa Precision và Recall. |
| 32 | ROC | Receiver Operating Characteristic | Đường cong biểu diễn quan hệ giữa tỷ lệ dương tính thật và dương tính giả. |
| 33 | AUC | Area Under the Curve | Diện tích dưới đường cong ROC, phản ánh khả năng phân biệt giữa các lớp. |
| 34 | Confusion Matrix | Ma trận nhầm lẫn | Bảng thể hiện số lượng mẫu được phân loại đúng và sai theo từng lớp. |
| 35 | SHAP | SHapley Additive exPlanations | Phương pháp giải thích mô hình dựa trên mức độ đóng góp của từng đặc trưng. |
| 36 | IDS | Intrusion Detection System | Hệ thống phát hiện xâm nhập, chủ yếu giám sát và cảnh báo hành vi bất thường. |
| 37 | IPS | Intrusion Prevention System | Hệ thống ngăn chặn xâm nhập, có khả năng phát hiện và thực hiện hành động phản ứng. |
| 38 | Firewall | Tường lửa | Cơ chế kiểm soát lưu lượng ra/vào hệ thống dựa trên tập luật bảo mật. |
| 39 | API | Application Programming Interface | Giao diện lập trình ứng dụng, cho phép các thành phần phần mềm giao tiếp với nhau. |

## 2. Mục Lục Dự Kiến

### Chương 1. Mở Đầu

**1.1. Lý do chọn đề tài**  
1.1.1. Bối cảnh phát triển của hệ thống mạng và dịch vụ trực tuyến  
1.1.2. Mức độ nguy hiểm của tấn công DDoS  
1.1.3. Sự cần thiết của hướng tiếp cận phát hiện DDoS bằng học máy  

**1.2. Mục tiêu nghiên cứu**  
1.2.1. Mục tiêu tổng quát  
1.2.2. Mục tiêu cụ thể  
1.2.3. Kết quả mong đợi của đề tài  

**1.3. Đối tượng và phạm vi nghiên cứu**  
1.3.1. Đối tượng nghiên cứu  
1.3.2. Phạm vi dữ liệu và loại tấn công  
1.3.3. Giới hạn triển khai của đề tài  

### Chương 2. Cơ Sở Lý Thuyết

**2.1. Tổng quan về tấn công DDoS**  
2.1.1. Khái niệm DoS và DDoS  
2.1.2. Đặc điểm của tấn công DDoS  
2.1.3. Ảnh hưởng của DDoS đối với hệ thống thông tin  

**2.2. Các dạng tấn công DDoS phổ biến**  
2.2.1. TCP SYN Flood  
2.2.2. UDP Flood và UDP-Lag Flood  
2.2.3. LDAP Flood, MSSQL Flood và NetBIOS Flood  

**2.3. Dữ liệu mạng và đặc trưng flow**  
2.3.1. Packet, PCAP và network flow  
2.3.2. Các nhóm đặc trưng thống kê của flow  
2.3.3. Ý nghĩa của đặc trưng flow trong phát hiện DDoS  

**2.4. Cơ sở lý thuyết về học máy**  
2.4.1. Học có giám sát trong bài toán phân loại  
2.4.2. Bài toán phân loại đa lớp  
2.4.3. Vấn đề mất cân bằng dữ liệu trong an toàn thông tin  

**2.5. Các thuật toán sử dụng trong đề tài**  
2.5.1. Random Forest và Extra Trees  
2.5.2. K-Nearest Neighbors và MLP Classifier  
2.5.3. XGBoost  

**2.6. Các thước đo đánh giá mô hình**  
2.6.1. Accuracy, Precision và Recall  
2.6.2. F1-Score, Macro F1 và ROC-AUC  
2.6.3. Confusion matrix và classification report  

### Chương 3. Phân Tích Và Thiết Kế Hệ Thống

**3.1. Yêu cầu hệ thống**  
3.1.1. Yêu cầu chức năng  
3.1.2. Yêu cầu phi chức năng  
3.1.3. Yêu cầu về dữ liệu và đánh giá mô hình  

**3.2. Kiến trúc tổng thể của hệ thống**  
3.2.1. Kiến trúc các thành phần chính  
3.2.2. Vai trò của backend, frontend và thư mục kết quả  
3.2.3. Mối liên hệ giữa huấn luyện offline và giám sát thời gian thực  

**3.3. Thiết kế pipeline dữ liệu**  
3.3.1. Nạp dữ liệu từ bộ CICDDoS2019  
3.3.2. Chuẩn hóa nhãn và lọc lớp dữ liệu  
3.3.3. Làm sạch dữ liệu và xử lý giá trị không hợp lệ  

**3.4. Thiết kế pipeline huấn luyện mô hình**  
3.4.1. Tiền xử lý chống rò rỉ dữ liệu  
3.4.2. Cross-validation và tối ưu siêu tham số  
3.4.3. Lựa chọn mô hình dựa trên khả năng tổng quát hóa  

**3.5. Thiết kế module giám sát và phản ứng**  
3.5.1. Thu thập lưu lượng bằng Scapy  
3.5.2. Gom packet thành flow và trích xuất đặc trưng  
3.5.3. Phát hiện, cảnh báo và hỗ trợ chặn IP bằng firewall  

### Chương 4. Thực Nghiệm Và Đánh Giá

**4.1. Môi trường thực nghiệm**  
4.1.1. Cấu hình phần cứng và phần mềm  
4.1.2. Ngôn ngữ lập trình và thư viện sử dụng  
4.1.3. Cấu trúc thư mục và dữ liệu đầu vào  

**4.2. Bộ dữ liệu CICDDoS2019**  
4.2.1. Mô tả bộ dữ liệu  
4.2.2. Các lớp dữ liệu sử dụng trong đề tài  
4.2.3. Phân tích phân phối nhãn và mất cân bằng dữ liệu  

**4.3. Quá trình huấn luyện và kiểm thử**  
4.3.1. Tiền xử lý dữ liệu  
4.3.2. Cấu hình mô hình và siêu tham số  
4.3.3. Kịch bản đánh giá mô hình  

**4.4. Kết quả thực nghiệm**  
4.4.1. Kết quả cross-validation  
4.4.2. Kết quả trên tập kiểm thử  
4.4.3. So sánh hiệu năng giữa các mô hình  

**4.5. Phân tích chi tiết kết quả**  
4.5.1. Phân tích confusion matrix  
4.5.2. Phân tích per-class metrics  
4.5.3. Phân tích overfitting và distribution shift  

### Chương 5. Kết Luận Và Hướng Phát Triển

**5.1. Kết quả đạt được**  
5.1.1. Kết quả về mặt nghiên cứu  
5.1.2. Kết quả về mặt triển khai hệ thống  
5.1.3. Kết quả về mặt đánh giá mô hình  

**5.2. Hạn chế của đề tài**  
5.2.1. Hạn chế về dữ liệu  
5.2.2. Hạn chế về khả năng tổng quát hóa  
5.2.3. Hạn chế của module phát hiện thời gian thực  

**5.3. Hướng phát triển**  
5.3.1. Mở rộng bộ dữ liệu và kịch bản tấn công  
5.3.2. Cải tiến trích xuất đặc trưng flow trong môi trường thực tế  
5.3.3. Tích hợp với hệ thống giám sát mạng thực tế  

## 3. Danh Mục Hình Ảnh Dự Kiến

| STT | Tên hình dự kiến | Nội dung thể hiện |
|---:|---|---|
| 1 | Hình 1.1. Mô hình tổng quan tấn công DDoS | Minh họa nhiều nguồn tấn công gửi lưu lượng đến máy chủ mục tiêu. |
| 2 | Hình 2.1. Phân loại các dạng tấn công DDoS | Sơ đồ nhóm tấn công DDoS theo giao thức hoặc hành vi lưu lượng. |
| 3 | Hình 2.2. Quá trình bắt tay TCP và cơ chế SYN Flood | Minh họa cách SYN Flood khai thác TCP handshake. |
| 4 | Hình 2.3. Quan hệ giữa packet, flow và dataset | Mô tả quá trình chuyển từ gói tin sang flow và đặc trưng dữ liệu. |
| 5 | Hình 3.1. Kiến trúc tổng thể hệ thống phát hiện DDoS | Trình bày các khối data loader, preprocessing, training, evaluation, dashboard và live IPS. |
| 6 | Hình 3.2. Luồng xử lý dữ liệu trong hệ thống | Minh họa quá trình từ dữ liệu Parquet đến mô hình đã huấn luyện. |
| 7 | Hình 3.3. Pipeline huấn luyện chống rò rỉ dữ liệu | Thể hiện preprocessing nằm trong pipeline và được fit theo từng fold. |
| 8 | Hình 3.4. Sơ đồ hoạt động của module live IPS | Minh họa quá trình bắt packet, gom flow, dự đoán và cảnh báo. |
| 9 | Hình 4.1. Phân phối số lượng mẫu theo lớp trong tập huấn luyện | Đồ thị cột thể hiện class imbalance ở tập train. |
| 10 | Hình 4.2. Phân phối số lượng mẫu theo lớp trong tập kiểm thử | Đồ thị cột thể hiện phân phối nhãn ở tập test. |
| 11 | Hình 4.3. Biểu đồ so sánh kết quả cross-validation giữa các mô hình | So sánh Accuracy, F1-score hoặc Macro F1 theo mô hình. |
| 12 | Hình 4.4. Biểu đồ so sánh kết quả kiểm thử giữa các mô hình | So sánh hiệu năng mô hình trên tập test. |
| 13 | Hình 4.5. Confusion matrix của mô hình tốt nhất | Thể hiện lỗi phân loại giữa các lớp. |
| 14 | Hình 4.6. ROC curve của mô hình tốt nhất | Minh họa khả năng phân biệt giữa các lớp. |
| 15 | Hình 4.7. Biểu đồ feature importance của mô hình cây | Thể hiện các đặc trưng có ảnh hưởng lớn nhất. |
| 16 | Hình 4.8. Biểu đồ SHAP summary | Giải thích đóng góp của các đặc trưng đối với dự đoán. |
| 17 | Hình 4.9. Biểu đồ train-CV-test F1-score | Minh họa khoảng cách overfitting và khả năng tổng quát hóa. |
| 18 | Hình 4.10. Learning curve của mô hình được chọn | Thể hiện quan hệ giữa số lượng mẫu huấn luyện và hiệu năng. |
| 19 | Hình 4.11. Giao diện dashboard phân tích kết quả | Ảnh chụp màn hình dashboard hiển thị metric và biểu đồ. |
| 20 | Hình 4.12. Giao diện giám sát live IPS | Ảnh chụp màn hình luồng sự kiện cảnh báo tấn công. |

## 4. Danh Mục Bảng Biểu Dự Kiến

| STT | Tên bảng dự kiến | Nội dung thể hiện |
|---:|---|---|
| 1 | Bảng 1.1. Bảng thuật ngữ và chữ viết tắt | Tổng hợp các thuật ngữ chính trong DDoS và học máy. |
| 2 | Bảng 2.1. So sánh DoS và DDoS | Phân biệt tấn công tập trung và tấn công phân tán. |
| 3 | Bảng 2.2. Đặc điểm các dạng tấn công DDoS sử dụng trong đề tài | Mô tả SYN Flood, UDP Flood, LDAP Flood, MSSQL Flood, NetBIOS Flood. |
| 4 | Bảng 2.3. Nhóm đặc trưng flow và ý nghĩa | Phân nhóm đặc trưng theo số lượng gói tin, kích thước, tốc độ, thời gian và cờ TCP. |
| 5 | Bảng 2.4. So sánh các thuật toán học máy sử dụng | Nêu ưu điểm, hạn chế và lý do lựa chọn RF, Extra Trees, KNN, MLP, XGBoost. |
| 6 | Bảng 3.1. Yêu cầu chức năng của hệ thống | Liệt kê các chức năng nạp dữ liệu, huấn luyện, đánh giá, dashboard và phát hiện live. |
| 7 | Bảng 3.2. Vai trò các module trong mã nguồn | Mô tả `data_loader.py`, `preprocessor.py`, `models.py`, `live_ips.py`, `mitigation.py`, `frontend/app.py`. |
| 8 | Bảng 4.1. Thống kê bộ dữ liệu CICDDoS2019 sử dụng trong đề tài | Số lượng mẫu, số đặc trưng và số lớp. |
| 9 | Bảng 4.2. Phân phối nhãn trong tập huấn luyện | Số lượng và tỷ lệ phần trăm từng lớp trong tập train. |
| 10 | Bảng 4.3. Phân phối nhãn trong tập kiểm thử | Số lượng và tỷ lệ phần trăm từng lớp trong tập test. |
| 11 | Bảng 4.4. Không gian siêu tham số của các mô hình | Trình bày các tham số được tối ưu trong RandomizedSearchCV. |
| 12 | Bảng 4.5. Kết quả cross-validation của các mô hình | Accuracy, F1-score mean/std, Macro F1 theo từng mô hình. |
| 13 | Bảng 4.6. Kết quả kiểm thử của các mô hình | Accuracy, Precision, Recall, F1-score, Macro F1 và ROC-AUC trên tập test. |
| 14 | Bảng 4.7. Per-class metrics của mô hình tốt nhất | Precision, Recall, F1-score và support theo từng lớp. |
| 15 | Bảng 4.8. So sánh train, validation và test metrics | Phân tích overfitting và generalization gap. |
| 16 | Bảng 4.9. Top đặc trưng quan trọng theo feature importance | Danh sách các đặc trưng có điểm quan trọng cao nhất. |
| 17 | Bảng 5.1. Tổng hợp kết quả đạt được so với mục tiêu đề tài | Đối chiếu mục tiêu ban đầu với kết quả thực hiện. |
| 18 | Bảng 5.2. Hạn chế và hướng phát triển | Tổng hợp các hạn chế hiện tại và đề xuất cải tiến. |
