# Model Creation Slide Note

## 1. Bài toán

Mục tiêu của mô hình là phân loại hành vi từ webcam thành `3 lớp`:

- `normal`
- `suspicious_action`
- `device`

Ngoài mô hình temporal, hệ thống runtime còn có:

- `YOLO` để phát hiện `phone`
- `rule-based absence check` để phát hiện `absence/offscreen`

Vì vậy, mô hình `LSTM` là phần lõi để phân loại hành vi theo thời gian, còn quyết định cuối cùng được tạo sau bước merge.

---

## 2. Nguồn dữ liệu

Dataset dùng để train:

- [oep_webcam_temporal_v3.jsonl](/Users/huy.dao/XuLyAnh/anti-cheat-demo/training/data/processed/oep_webcam_temporal_v3.jsonl)

Dataset gốc:

- OEP dataset của Michigan State University
- `24 subjects`
- mỗi subject có:
  - `1 webcam video`
  - `1 wearcam video`
  - `1 audio file`
  - `1 gt.txt`

Sau khi parse:

- `565` segment có nhãn từ `gt.txt`
- suy thêm `262` segment `normal` từ khoảng trống giữa các event
- tổng cộng: `827 samples`

Lưu ý:

- `24 subjects` không có nghĩa là chỉ có `24 sample`
- mỗi subject có nhiều segment
- mỗi segment sau khi chuyển thành sequence sẽ trở thành `1 sample train`

---

## 3. Cách chia dữ liệu

Dataset được chia theo `subject`, không chia ngẫu nhiên theo sample.

Lý do:

- tránh việc cùng một người xuất hiện ở cả train và test
- giảm nguy cơ overestimate kết quả

Kết quả chia hiện tại:

- `train = 518`
- `val = 167`
- `test = 142`

Phân bố lớp:

- Train
  - `suspicious_action = 329`
  - `normal = 186`
  - `device = 3`
- Validation
  - `suspicious_action = 90`
  - `normal = 77`
  - `device = 0`
- Test
  - `suspicious_action = 79`
  - `normal = 51`
  - `device = 12`

Hạn chế:

- lớp `device` rất ít
- tập `validation` không có `device`

---

## 4. Quy trình build sample train

Từ mỗi segment của OEP:

1. đọc thời gian bắt đầu và kết thúc từ `gt.txt`
2. map nhãn gốc của OEP sang nhãn v3
3. lấy `16` timestamp chia đều trong đoạn đó
4. đọc `16` frame từ webcam video
5. mỗi frame được chuyển thành `19 features`
6. gom lại thành ma trận `16 x 19`
7. lưu thành `1 sample` trong file `.jsonl`

Nói ngắn:

- `1 segment` -> `16 frame samples` -> `19 features / frame` -> `1 temporal sample`

---

## 5. Map nhãn của OEP sang nhãn v3

Nhãn OEP gốc được rút về `3 lớp` để phù hợp với demo:

- `normal -> normal`
- `type_1 -> suspicious_action`
- `type_2 -> suspicious_action`
- `type_3 -> normal`
- `type_5 -> device`
- `type_6 -> suspicious_action`

Ý nghĩa:

- phần lớn hành vi bất thường được gom vào `suspicious_action`
- `type_5` được giữ riêng cho `device`
- `type_3` hiện được xem là `normal` trong bối cảnh single-camera demo

---

## 6. 19 features của mỗi frame

Nguồn code:

- [feature_extractor.py](/Users/huy.dao/XuLyAnh/anti-cheat-demo/backend/oep_service/feature_extractor.py)

Danh sách feature:

1. `brightness`
2. `motion_score`
3. `edge_density`
4. `face_present`
5. `face_area_ratio`
6. `face_center_x`
7. `face_center_y`
8. `eye_pair_present`
9. `eye_distance_ratio`
10. `yaw_proxy`
11. `pitch_proxy`
12. `eye_open_ratio`
13. `lower_face_edge_density`
14. `multiple_faces_proxy`
15. `upperbody_present`
16. `upperbody_area_ratio`
17. `upperbody_center_x`
18. `upperbody_center_y`
19. `face_body_relation`

Ý nghĩa chung:

- nhóm `image-level`: ánh sáng, chuyển động, mật độ cạnh
- nhóm `face`: sự hiện diện và vị trí khuôn mặt
- nhóm `eye`: mắt, hướng nhìn gần đúng, độ mở mắt
- nhóm `body`: upper-body và quan hệ giữa mặt với thân người

---

## 7. Kiến trúc mô hình

Model train chính:

- `LSTM temporal classifier`

Kiến trúc:

- `LSTM(input_size=19, hidden_size=64, batch_first=True)`
- classifier head:
  - `Linear(64 -> 64)`
  - `ReLU`
  - `Dropout(0.2)`
  - `Linear(64 -> 3)`

Ý tưởng:

- mô hình không nhìn từng ảnh đơn
- mô hình nhìn một chuỗi feature theo thời gian
- phù hợp với bài toán hành vi, vì cheating là pattern theo chuỗi frame

---

## 8. Quy trình train

Script train:

- [train_temporal_model.py](/Users/huy.dao/XuLyAnh/anti-cheat-demo/training/scripts/train_temporal_model.py)

Các bước train:

1. đọc dataset `.jsonl`
2. tách `train / val / test`
3. tính `feature_mean` và `feature_std` trên tập `train`
4. chuẩn hóa toàn bộ sequence theo thống kê của train
5. dùng `WeightedRandomSampler` để giảm mất cân bằng lớp
6. train `LSTM` nhiều epoch
7. chọn `best_model.pt` theo `best_val_accuracy`
8. tune `non_normal_threshold` trên validation
9. evaluate trên test
10. lưu:
   - `best_model.pt`
   - `metrics.json`
   - `label_map.json`

---

## 9. Tham số train chính

Thông số của model hiện đang dùng:

- `epochs = 24`
- `batch_size = 16`
- `hidden_size = 64`
- `learning_rate = 0.0007`
- `weight_decay = 0.0001`
- `feature_dim = 19`
- `num_classes = 3`
- `max_frames = 90`
- `balanced_sampler = true`

Loss và optimizer:

- `CrossEntropyLoss`
- `Adam`

Class weights:

- `device = 7.5865`
- `normal = 0.9635`
- `suspicious_action = 0.7244`

Lý do:

- lớp `device` rất hiếm
- cần tăng trọng số để model không bỏ qua lớp này

---

## 10. Cách suy luận ở runtime

Khi frontend gửi lên một frame:

1. gọi `extract_frame_features(...)`
2. build vector `19 features`
3. append vào `feature_buffer`
4. nếu buffer dưới `8` frame:
   - chưa predict
   - trả trạng thái `collecting temporal sequence`
5. nếu buffer từ `8` frame trở lên:
   - lấy sequence hiện có trong buffer
   - normalize bằng `feature_mean` và `feature_std`
   - đưa qua `LSTM`
   - lấy `logits`
   - chạy `softmax`
   - chọn lớp `top-1`
   - nếu lớp khác `normal` nhưng confidence thấp hơn `non_normal_threshold = 0.51` thì ép về `normal`

Thông số runtime:

- `SEQUENCE_FRAMES = 16`
- `MIN_FRAMES_TO_PREDICT = 8`

Nghĩa là:

- từ frame `8` bắt đầu dự đoán
- từ frame `16` trở đi luôn dùng `16 frame gần nhất`

---

## 11. Tại sao dùng softmax

`LSTM` trả ra `logits`, tức là điểm số thô cho từng lớp.

`Softmax` được dùng để:

- chuyển logits thành xác suất
- làm tổng xác suất bằng `1`
- dễ hiển thị lên UI
- dễ áp threshold
- dễ so sánh lớp nào mạnh nhất

Ví dụ:

- trước softmax: `[-0.4, 1.7, 2.1]`
- sau softmax:
  - `device = 0.05`
  - `suspicious_action = 0.40`
  - `normal = 0.55`

---

## 12. Ba work stream ở runtime

Runtime hiện có `3 luồng`:

1. `Temporal model`
   - phân loại `normal / suspicious_action / device`
2. `YOLO`
   - kiểm tra `phone`
3. `Absence rule`
   - kiểm tra mất cả `face` và `upper-body`

---

## 13. Merge decision hoạt động như thế nào

Input của merge:

- output từ temporal model:
  - `prediction_label`
  - `confidence`
  - `probabilities`
- output từ YOLO:
  - `device_active`
  - `device_confidence`
- output từ absence rule:
  - `absence_active`
  - `absence_confidence`

Thứ tự ưu tiên:

1. `device`
2. `absence/offscreen`
3. `normal guard`
4. output gốc từ temporal model

Ví dụ:

- temporal model:
  - `normal = 0.72`
  - `suspicious_action = 0.25`
  - `device = 0.03`
- nhưng YOLO thấy `phone = 0.80`

Sau merge:

- override nhãn cuối thành `device`
- scale lại các xác suất còn lại
- ví dụ kết quả mới gần đúng:
  - `device = 0.80`
  - `normal = 0.1485`
  - `suspicious_action = 0.0515`

Ý nghĩa:

- model temporal cho biết ngữ cảnh hành vi
- YOLO cho biết có thiết bị rõ ràng hay không
- merge dùng thứ tự ưu tiên để ra quyết định cuối cùng

---

## 14. Kết quả model hiện tại

Đọc từ:

- [metrics.json](/Users/huy.dao/XuLyAnh/anti-cheat-demo/training/models/oep_webcam_monitor_v3/metrics.json)

Kết quả chính:

- `best_val_accuracy = 73.65%`
- `val_accuracy_thresholded = 74.85%`
- `val_macro_f1_thresholded = 49.45%`
- `test_accuracy = 56.34%`
- `test_accuracy_thresholded = 55.63%`
- `test_macro_f1_thresholded = 38.87%`
- `non_normal_threshold = 0.51`

Nhận xét:

- mô hình đã học được phân biệt `normal` và `suspicious_action` ở mức dùng được cho demo
- lớp `device` vẫn cần hỗ trợ bởi `YOLO runtime`
- OEP không hoàn toàn khớp với điều kiện `1 camera laptop`, nên kết quả vẫn còn giới hạn

---

## 15. Đầu ra của mô hình theo số liệu

### 15.1 Đầu ra trực tiếp của temporal model

Mỗi lần buffer đủ dữ liệu, temporal model trả ra:

- `prediction_label`
- `confidence`
- `probabilities`

Ví dụ đầu ra:

```json
{
  "prediction_label": "normal",
  "confidence": 0.72,
  "probabilities": [
    {"label": "normal", "confidence": 0.72},
    {"label": "suspicious_action", "confidence": 0.25},
    {"label": "device", "confidence": 0.03}
  ]
}
```

Ý nghĩa:

- `prediction_label`: lớp được chọn sau bước `top-1` và threshold
- `confidence`: độ tin cậy của lớp cuối cùng
- `probabilities`: phân phối xác suất giữa các lớp

### 15.2 Đầu ra sau bước merge

Sau khi merge với `YOLO` và `absence rule`, backend trả về:

- `prediction_label`
- `confidence`
- `probabilities`
- `status_text`
- `annotated_frame`
- `features`

Ví dụ nếu temporal model nghiêng về `normal`, nhưng YOLO thấy phone:

```json
{
  "prediction_label": "device",
  "confidence": 0.80,
  "probabilities": [
    {"label": "device", "confidence": 0.80},
    {"label": "normal", "confidence": 0.1485},
    {"label": "suspicious_action", "confidence": 0.0515}
  ]
}
```

Ý nghĩa:

- temporal model đưa ra ngữ cảnh hành vi
- YOLO có thể override sang `device`
- merge sẽ scale lại các xác suất còn lại để tổng vẫn bằng `1`

### 15.3 Thống kê đầu ra trên tập validation

Theo `metrics.json`, số lượng prediction sau threshold trên tập `validation` là:

- `normal = 63`
- `suspicious_action = 104`
- `device = 0`

Prediction thô trước threshold:

- `normal = 60`
- `suspicious_action = 106`
- `device = 1`

### 15.4 Thống kê đầu ra trên tập test

Theo `metrics.json`, số lượng prediction sau threshold trên tập `test` là:

- `normal = 82`
- `suspicious_action = 59`
- `device = 1`

Prediction thô trước threshold:

- `normal = 81`
- `suspicious_action = 60`
- `device = 1`

### 15.5 Kết quả end-to-end đã kiểm tra qua API

Các case đã test:

- `frontal_repeat`
  - kết quả cuối: `normal`
  - confidence: `0.62`
  - probabilities:
    - `normal = 0.62`
    - `suspicious_action = 0.3787`
    - `device = 0.0013`

- `suspicious`
  - kết quả cuối: `suspicious_action`
  - confidence: `0.954`
  - probabilities:
    - `suspicious_action = 0.954`
    - `normal = 0.0441`
    - `device = 0.0019`

- `absence/offscreen`
  - khi mất cả `face` và `upper-body` liên tiếp
  - confidence runtime có thể tăng dần từ:
    - `0.55`
    - `0.60`
    - `0.65`
    - ...
    - đến tối đa `0.99`

### 15.6 Cách nói ngắn trên slide

- Output của model là `label + confidence + class probabilities`
- Output cuối của hệ thống là `label + confidence + probabilities + status + annotated frame`
- Validation tốt nhất hiện tại: `73.65%`
- Test accuracy sau threshold: `55.63%`

---

## 16. Câu chốt ngắn để đưa vào slide

> Hệ thống huấn luyện một mô hình `LSTM` trên chuỗi `16 frame`, mỗi frame được biểu diễn bằng `19 features` thủ công. Mô hình temporal dự đoán `normal`, `suspicious_action` hoặc `device`, sau đó kết quả được hợp nhất với `YOLO` và `absence rule` để tạo ra quyết định cuối cùng cho giao diện giám sát.
