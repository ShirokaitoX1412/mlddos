# Đồ Án Tốt Nghiệp

**Tên đề tài:** Nghiên cứu xây dựng giải pháp phát hiện DDoS sử dụng học máy

## Chương 1. Tổng Quan Đề Tài

### 1.1. Lý do chọn đề tài

Trong bối cảnh chuyển đổi số diễn ra mạnh mẽ, các hệ thống thông tin, dịch vụ trực tuyến, nền tảng thương mại điện tử, hệ thống ngân hàng số, cổng dịch vụ công và hạ tầng điện toán đám mây ngày càng giữ vai trò quan trọng trong hoạt động của tổ chức, doanh nghiệp và cá nhân. Cùng với sự phát triển đó, các nguy cơ mất an toàn thông tin cũng gia tăng cả về quy mô, tần suất và mức độ phức tạp. Trong số các hình thức tấn công mạng phổ biến, tấn công từ chối dịch vụ phân tán, hay Distributed Denial of Service (DDoS), là một trong những mối đe dọa nghiêm trọng đối với tính sẵn sàng của hệ thống.

Tấn công DDoS được thực hiện bằng cách huy động một số lượng lớn thiết bị hoặc nguồn phát tán lưu lượng để gửi đồng thời nhiều yêu cầu đến hệ thống mục tiêu. Mục tiêu của kẻ tấn công là làm cạn kiệt tài nguyên xử lý, băng thông mạng, bộ nhớ, hàng đợi kết nối hoặc tài nguyên ứng dụng, từ đó khiến dịch vụ bị gián đoạn hoặc không thể phục vụ người dùng hợp lệ. Khác với một số dạng tấn công tập trung vào đánh cắp dữ liệu hoặc chiếm quyền điều khiển, DDoS chủ yếu tác động đến thuộc tính sẵn sàng trong bộ ba an toàn thông tin CIA, bao gồm Confidentiality, Integrity và Availability. Tuy nhiên, hậu quả của DDoS có thể rất lớn, bao gồm thiệt hại tài chính, ảnh hưởng uy tín, gián đoạn hoạt động nghiệp vụ và làm suy giảm niềm tin của người dùng.

Các phương pháp phòng chống DDoS truyền thống thường dựa trên luật tĩnh, chữ ký, ngưỡng lưu lượng hoặc danh sách chặn IP. Những phương pháp này có ưu điểm là dễ triển khai và có thể phản ứng nhanh với các mẫu tấn công đã biết. Tuy nhiên, chúng cũng tồn tại nhiều hạn chế. Thứ nhất, lưu lượng mạng trong thực tế thường biến đổi theo thời gian, theo hành vi người dùng và theo đặc thù dịch vụ, khiến việc đặt ngưỡng tĩnh trở nên khó chính xác. Thứ hai, các kiểu tấn công DDoS hiện đại có thể thay đổi mẫu lưu lượng, giả mạo địa chỉ nguồn, phân tán qua botnet hoặc mô phỏng hành vi người dùng hợp lệ để né tránh các cơ chế phát hiện dựa trên luật. Thứ ba, khi hệ thống chỉ dựa vào chữ ký đã biết, khả năng phát hiện các biến thể tấn công mới thường bị hạn chế.

Trong bối cảnh đó, học máy trở thành một hướng tiếp cận có tiềm năng trong bài toán phát hiện DDoS. Thay vì chỉ dựa vào các luật được định nghĩa thủ công, mô hình học máy có thể học các mẫu hành vi từ dữ liệu lưu lượng mạng, từ đó phân biệt giữa lưu lượng bình thường và lưu lượng tấn công. Đặc biệt, với dữ liệu được biểu diễn ở mức flow, các mô hình học máy có thể khai thác nhiều đặc trưng thống kê như số lượng gói tin, kích thước gói tin, tốc độ truyền, khoảng thời gian giữa các gói tin và các cờ giao thức TCP. Đây là những tín hiệu có giá trị trong việc nhận diện hành vi bất thường của các cuộc tấn công DDoS.

Từ những lý do trên, đề tài “Nghiên cứu xây dựng giải pháp phát hiện DDoS sử dụng học máy” được lựa chọn nhằm nghiên cứu, xây dựng và đánh giá một pipeline phát hiện DDoS dựa trên dữ liệu flow. Đề tài không chỉ tập trung vào việc huấn luyện mô hình phân loại, mà còn hướng đến xây dựng một quy trình tương đối đầy đủ gồm nạp dữ liệu, tiền xử lý, huấn luyện, đánh giá, giải thích mô hình và mô phỏng khả năng giám sát/phản ứng trong môi trường thực tế.

### 1.2. Mục tiêu nghiên cứu

Mục tiêu tổng quát của đề tài là nghiên cứu và xây dựng một giải pháp phát hiện tấn công DDoS sử dụng học máy, có khả năng phân loại lưu lượng mạng thành lưu lượng bình thường và các nhóm tấn công DDoS phổ biến dựa trên đặc trưng flow.

Các mục tiêu cụ thể bao gồm:

- Nghiên cứu tổng quan về tấn công DDoS, các dạng DDoS phổ biến và những đặc điểm hành vi thể hiện qua lưu lượng mạng.
- Tìm hiểu cách biểu diễn dữ liệu mạng ở mức flow và ý nghĩa của các đặc trưng flow trong bài toán phát hiện DDoS.
- Xây dựng pipeline xử lý dữ liệu từ bộ dữ liệu CICDDoS2019, bao gồm nạp dữ liệu, chuẩn hóa nhãn, xử lý dữ liệu trùng lặp, xử lý giá trị không hợp lệ, chuẩn hóa đặc trưng và loại bỏ đặc trưng không phù hợp.
- Huấn luyện và so sánh nhiều thuật toán học máy, bao gồm Random Forest, Extra Trees, K-Nearest Neighbors, MLP Classifier và XGBoost.
- Áp dụng cross-validation, tối ưu siêu tham số và các kỹ thuật hạn chế overfitting nhằm cải thiện khả năng tổng quát hóa của mô hình trên dữ liệu chưa nhìn thấy.
- Đánh giá mô hình bằng nhiều thước đo khác nhau như Accuracy, Precision, Recall, F1-score, Macro F1, ROC-AUC, confusion matrix và classification report.
- Phân tích khả năng giải thích của mô hình thông qua feature importance và SHAP để xác định các đặc trưng có ảnh hưởng lớn đến kết quả phân loại.
- Xây dựng thành phần mô phỏng giám sát lưu lượng thời gian thực và cơ chế hỗ trợ giảm thiểu tấn công thông qua firewall.

Thông qua các mục tiêu trên, đề tài hướng đến việc xây dựng một giải pháp có tính thực nghiệm rõ ràng, có khả năng đánh giá khách quan hiệu quả của học máy trong bài toán phát hiện DDoS, đồng thời chỉ ra các hạn chế còn tồn tại khi triển khai trong môi trường thực tế.

### 1.3. Đối tượng và phạm vi nghiên cứu

Đối tượng nghiên cứu của đề tài là bài toán phát hiện và phân loại tấn công DDoS dựa trên dữ liệu lưu lượng mạng ở mức flow. Cụ thể, đề tài tập trung vào các đặc trưng thống kê được trích xuất từ luồng mạng, bao gồm thông tin về giao thức, thời lượng flow, số lượng gói tin, kích thước gói tin, tốc độ truyền, khoảng thời gian giữa các gói tin, số byte truyền nhận và các cờ TCP.

Bộ dữ liệu sử dụng trong đề tài là CICDDoS2019, một bộ dữ liệu phổ biến trong nghiên cứu phát hiện tấn công DDoS. Dữ liệu được lưu dưới định dạng Parquet, bao gồm các tập huấn luyện và kiểm thử cho nhiều loại tấn công khác nhau. Trong phạm vi triển khai của hệ thống, các lớp được sử dụng gồm lưu lượng bình thường và một số dạng tấn công DDoS như TCP SYN Flood, UDP Flood, UDP-Lag Flood, LDAP Flood, MSSQL Flood và NetBIOS Flood.

Phạm vi nghiên cứu của đề tài bao gồm:

- Nghiên cứu dữ liệu flow và đặc trưng thống kê phục vụ phát hiện DDoS.
- Xây dựng pipeline huấn luyện mô hình học máy trên dữ liệu CICDDoS2019.
- So sánh hiệu quả của nhiều thuật toán phân loại khác nhau.
- Phân tích kết quả đánh giá trên tập huấn luyện, cross-validation và tập kiểm thử.
- Xây dựng dashboard hiển thị kết quả và module mô phỏng phát hiện lưu lượng thời gian thực.

Đề tài không đi sâu vào phân tích payload của gói tin, không xây dựng hệ thống chống DDoS ở quy mô nhà mạng và không triển khai thử nghiệm trên hạ tầng production thực tế. Thành phần phát hiện thời gian thực trong đồ án được xem là nguyên mẫu minh họa cách mô hình có thể được tích hợp vào hệ thống giám sát, chưa thay thế hoàn toàn các công cụ phát hiện và giảm thiểu DDoS chuyên dụng trong môi trường doanh nghiệp.

### 1.4. Phương pháp nghiên cứu

Đề tài sử dụng kết hợp phương pháp nghiên cứu lý thuyết và phương pháp thực nghiệm.

Về mặt lý thuyết, đề tài nghiên cứu các khái niệm nền tảng liên quan đến tấn công DDoS, phát hiện xâm nhập, dữ liệu mạng mức packet và flow, các thuật toán học máy phục vụ phân loại, cũng như các thước đo đánh giá mô hình. Việc nghiên cứu lý thuyết giúp xác định cơ sở khoa học cho việc lựa chọn đặc trưng, lựa chọn mô hình và thiết kế pipeline phát hiện.

Về mặt thực nghiệm, đề tài sử dụng bộ dữ liệu CICDDoS2019 để xây dựng và đánh giá mô hình. Quy trình thực nghiệm bao gồm các bước chính: nạp dữ liệu, chuẩn hóa nhãn, làm sạch dữ liệu, tiền xử lý đặc trưng, huấn luyện mô hình, tối ưu siêu tham số, đánh giá kết quả và phân tích lỗi. Các mô hình được đánh giá không chỉ bằng độ chính xác tổng thể mà còn bằng các thước đo chi tiết hơn như Precision, Recall, F1-score, Macro F1 và confusion matrix. Điều này giúp hạn chế việc đánh giá thiên lệch trong trường hợp dữ liệu mất cân bằng giữa các lớp.

Bên cạnh đó, đề tài áp dụng cross-validation để đánh giá độ ổn định của mô hình trên nhiều phân hoạch dữ liệu khác nhau. Các bước tiền xử lý được đặt trong pipeline nhằm đảm bảo scaler, imputer và feature selector chỉ được fit trên dữ liệu huấn luyện của từng fold, từ đó giảm nguy cơ rò rỉ dữ liệu. Đối với các mô hình có nhiều siêu tham số, đề tài sử dụng RandomizedSearchCV để tìm kiếm cấu hình phù hợp trong không gian tham số.

Ngoài việc đánh giá hiệu năng, đề tài còn sử dụng feature importance và SHAP để phân tích khả năng giải thích của mô hình. Việc này giúp làm rõ các đặc trưng nào có ảnh hưởng lớn đến quá trình phân loại, đồng thời tăng tính minh bạch của giải pháp học máy trong bối cảnh an toàn thông tin.



### 1.5. Ý nghĩa khoa học và thực tiễn

Về mặt khoa học, đề tài góp phần làm rõ khả năng ứng dụng học máy trong bài toán phát hiện DDoS dựa trên dữ liệu flow. Thay vì chỉ tiếp cận bài toán theo hướng luật tĩnh hoặc chữ ký, đề tài khai thác các đặc trưng thống kê của luồng mạng và sử dụng mô hình học máy để nhận diện mẫu hành vi bất thường. Việc so sánh nhiều thuật toán khác nhau giúp đánh giá ưu điểm và hạn chế của từng nhóm mô hình trong cùng một bối cảnh dữ liệu.

Đề tài cũng nhấn mạnh tầm quan trọng của quy trình đánh giá khách quan. Trong các bài toán an toàn thông tin, việc mô hình đạt kết quả cao trên tập validation chưa đủ để kết luận mô hình có khả năng hoạt động tốt trong thực tế. Sự khác biệt giữa phân phối dữ liệu huấn luyện và dữ liệu kiểm thử có thể làm hiệu năng suy giảm đáng kể. Do đó, việc phân tích overfitting, class imbalance, per-class metrics và khả năng tổng quát hóa là cần thiết để đánh giá đúng chất lượng mô hình.

Về mặt thực tiễn, hệ thống được xây dựng có thể được sử dụng như một nguyên mẫu cho giải pháp phát hiện DDoS trong môi trường nghiên cứu hoặc phòng thí nghiệm. Pipeline huấn luyện giúp tự động hóa quá trình xử lý dữ liệu, huấn luyện và đánh giá mô hình. Dashboard hỗ trợ trực quan hóa kết quả, giúp người vận hành hoặc người nghiên cứu dễ dàng theo dõi hiệu năng mô hình. Module live IPS và mitigation minh họa cách tích hợp mô hình học máy vào quy trình giám sát và phản ứng trước tấn công.

Ngoài ra, đề tài có giá trị tham khảo cho sinh viên và người nghiên cứu khi xây dựng các hệ thống phát hiện xâm nhập dựa trên học máy. Các nội dung như chống data leakage, xử lý mất cân bằng dữ liệu, cross-validation, phân tích confusion matrix và giải thích mô hình là những yếu tố quan trọng trong quá trình phát triển hệ thống học máy có khả năng ứng dụng thực tế.

### 1.6. Cấu trúc đồ án

Nội dung đồ án được tổ chức thành năm chương chính.

**Chương 1. Tổng quan đề tài** trình bày lý do chọn đề tài, mục tiêu nghiên cứu, đối tượng và phạm vi nghiên cứu, phương pháp nghiên cứu, ý nghĩa khoa học và thực tiễn của đề tài.

**Chương 2. Cơ sở lý thuyết** trình bày các kiến thức nền tảng về tấn công DDoS, các dạng DDoS phổ biến, dữ liệu mạng mức packet và flow, các đặc trưng phục vụ phát hiện DDoS, các thuật toán học máy được sử dụng và các thước đo đánh giá mô hình phân loại.

**Chương 3. Phân tích và thiết kế hệ thống** mô tả yêu cầu của hệ thống, kiến trúc tổng thể, luồng dữ liệu, pipeline xử lý dữ liệu, pipeline huấn luyện mô hình, module phát hiện thời gian thực, dashboard giám sát và cơ chế hỗ trợ giảm thiểu tấn công.

**Chương 4. Xây dựng và huấn luyện mô hình** trình bày chi tiết bộ dữ liệu CICDDoS2019, quá trình tiền xử lý, lựa chọn đặc trưng, xây dựng pipeline chống rò rỉ dữ liệu, huấn luyện các mô hình học máy, tối ưu siêu tham số, xử lý mất cân bằng dữ liệu và giải thích mô hình.

**Chương 5. Thực nghiệm, đánh giá và kết luận** trình bày môi trường thực nghiệm, kịch bản kiểm thử, kết quả đánh giá theo từng mô hình, phân tích confusion matrix và per-class metrics, đánh giá khả năng tổng quát hóa, nêu các hạn chế còn tồn tại, hướng phát triển và kết luận chung của đồ án.
