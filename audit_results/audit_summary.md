# Báo cáo kiểm tra đánh giá mô hình

## Kết luận nhanh

- Pipeline đã được chỉnh để chia stratified train/validation trước khi fit imputer, chọn đặc trưng và scaler.
- Train và validation gần nhau, nhưng test thấp hơn nhiều; dấu hiệu chính là phân phối lớp giữa train và test bị lệch mạnh.
- Nên đọc thêm `per_class_metrics.csv` vì weighted average che mất các lớp hiếm như `UDP-Lag Flood`.

## Kiểm tra split/leakage

- Stratified split: `True`
- Cột single-value bị loại: `12`
- Cột tương quan cao bị loại: `34`
- Dòng feature trùng train/validation: `2241`
- Dòng feature trùng train/test: `4054`
- Dòng feature trùng validation/test: `1031`

## Lớp lệch phân phối mạnh nhất

| Lớp | Train % | Test % | Chênh lệch |
|-----|---------|--------|------------|
| TCP SYN Flood | 40.4263 | 1.3780 | 39.0483 |
| UDP-Lag Flood | 0.0471 | 22.9369 | 22.8898 |
| UDP Flood | 15.2263 | 26.9390 | 11.7127 |
| MSSQL Flood | 7.2164 | 16.0600 | 8.8436 |
| Benign | 35.0646 | 27.4173 | 7.6473 |
| LDAP Flood | 1.6129 | 3.7229 | 2.1100 |
| NetBIOS Flood | 0.4064 | 1.5460 | 1.1396 |

## Khoảng cách validation-test

| Mô hình | Train F1 | Validation F1 | Test F1 | Gap Val-Test |
|---------|----------|---------------|---------|--------------|
| Extra Trees | 0.992737 | 0.991793 | 0.642486 | 0.349307 |
| KNN | 0.996438 | 0.992200 | 0.712075 | 0.280125 |
| MLP Classifier | 0.988092 | 0.988331 | 0.708714 | 0.279617 |
| Random Forest | 0.994967 | 0.993578 | 0.646413 | 0.347165 |
| XGBoost | 0.993961 | 0.992277 | 0.709566 | 0.282711 |

## Cross-validation

| Mô hình | CV F1 weighted mean | CV F1 weighted std | CV F1 macro mean |
|---------|---------------------|--------------------|------------------|
| Random Forest | 0.992997 | 0.000326 | 0.872737 |
| KNN | 0.991890 | 0.000323 | 0.876145 |
| Extra Trees | 0.991495 | 0.000382 | 0.835397 |
| MLP Classifier | 0.988291 | 0.000083 | 0.817634 |
| XGBoost | 0.991886 | 0.000234 | 0.868920 |

## Hướng kiểm tra tiếp theo

- Huấn luyện thêm biến thể `class_weight='balanced'` cho Random Forest/Extra Trees.
- Thử oversampling/undersampling cho các lớp hiếm, đặc biệt `UDP-Lag Flood`.
- Tách validation theo thời gian hoặc theo file/tấn công để mô phỏng test thực tế hơn.
- Không chỉ tối ưu F1 weighted; theo dõi thêm macro F1 và per-class recall.
