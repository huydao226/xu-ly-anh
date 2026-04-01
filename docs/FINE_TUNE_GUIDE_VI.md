# Hướng Dẫn Fine-Tune

## 1. Mục tiêu

Tài liệu này mô tả cách cung cấp dữ liệu và huấn luyện thêm AI cho đề tài giám sát gian lận thi trực tuyến bằng 1 camera laptop.

Sau khi làm xong các bước dưới đây, project sẽ có 2 phần training rõ ràng:

- fine-tune `YOLO` cho vật thể gian lận
- train `LSTM` cho hành vi gian lận theo thời gian

## 2. Dữ liệu cần chuẩn bị

### 2.1 Dataset chính

Dataset chính nên là dữ liệu tự quay từ người Việt Nam hoặc người châu Á trong cùng bối cảnh sử dụng laptop webcam như lúc demo.

Nên quay tối thiểu:

- 20 đến 30 người
- nhiều điều kiện ánh sáng
- có người đeo kính và không đeo kính
- nhiều góc đặt laptop khác nhau
- khoảng cách ngồi gần và xa khác nhau

### 2.2 Tình huống cần quay

Phải có cả `negative cases` và `positive cases`.

Negative cases:

- ngồi làm bài bình thường
- chớp mắt
- chỉnh ghế
- ngả người ra sau rồi ngồi lại
- quay đầu rất ngắn rồi quay lại
- cúi xuống ngắn để chỉnh tư thế

Positive cases:

- nhìn sang trái lâu
- nhìn sang phải lâu
- cúi xuống đọc tài liệu
- đưa điện thoại vào khung hình
- để tài liệu trong khung hình
- có người thứ hai xuất hiện
- rời khỏi khung hình

## 3. Cách lấy dữ liệu ngay từ hệ thống hiện tại

Project đã có sẵn chức năng lưu session làm dữ liệu thô.

Mở file:

```text
/Users/huy.dao/XuLyAnh/anti-cheat-demo/backend/.env
```

Sửa thành:

```bash
DATASET_CAPTURE_ENABLED=true
DATASET_CAPTURE_ROOT=/Users/huy.dao/XuLyAnh/anti-cheat-demo/training/data/raw_sessions
```

Sau đó restart backend và chạy demo bình thường.

Mỗi session sẽ được lưu tại:

```text
/Users/huy.dao/XuLyAnh/anti-cheat-demo/training/data/raw_sessions/<session_id>/
```

Trong đó có:

- `images/`: frame JPG đã cắt từ webcam
- `metadata.jsonl`: metadata từng frame, gồm metric và event hiện tại
- `session_manifest.json`: metadata của session

## 4. Bạn cần cung cấp dữ liệu fine-tune như thế nào

### 4.1 Fine-tune YOLO

Bạn cần tạo file:

```text
/Users/huy.dao/XuLyAnh/anti-cheat-demo/training/data/annotations/object_annotations.csv
```

Schema:

- `image_path`
- `label`
- `x_min`
- `y_min`
- `x_max`
- `y_max`
- `split`

Ví dụ mẫu nằm ở:

```text
/Users/huy.dao/XuLyAnh/anti-cheat-demo/training/data/annotations/object_annotations.sample.csv
```

Label nên dùng:

- `phone`
- `notes`
- `second_person`
- `calculator`

### 4.2 Train temporal model

Bạn cần tạo file:

```text
/Users/huy.dao/XuLyAnh/anti-cheat-demo/training/data/annotations/behavior_segments.csv
```

Schema:

- `session_id`
- `start_frame`
- `end_frame`
- `label`
- `split`
- `notes`

Ví dụ mẫu nằm ở:

```text
/Users/huy.dao/XuLyAnh/anti-cheat-demo/training/data/annotations/behavior_segments.sample.csv
```

Label nên dùng:

- `normal`
- `look_away`
- `looking_down`
- `phone_use`
- `multiple_faces`

## 5. Cách build dataset

### 5.1 Build YOLO dataset

```bash
cd /Users/huy.dao/XuLyAnh/anti-cheat-demo
python3 training/scripts/prepare_yolo_dataset.py \
  --annotations training/data/annotations/object_annotations.csv \
  --output-dir training/data/processed/yolo_exam_v1
```

### 5.2 Build temporal dataset

```bash
cd /Users/huy.dao/XuLyAnh/anti-cheat-demo
python3 training/scripts/build_temporal_dataset.py \
  --sessions-root training/data/raw_sessions \
  --labels training/data/annotations/behavior_segments.csv \
  --output training/data/processed/temporal_sequences.jsonl
```

## 6. Cách train

### 6.1 Train YOLO

```bash
cd /Users/huy.dao/XuLyAnh/anti-cheat-demo
yolo task=detect mode=train \
  model=backend/yolov8n.pt \
  data=training/data/processed/yolo_exam_v1/data.yaml \
  epochs=50 \
  imgsz=640 \
  project=training/models \
  name=yolo_exam_v1
```

### 6.2 Train LSTM

```bash
cd /Users/huy.dao/XuLyAnh/anti-cheat-demo
python3 training/scripts/train_temporal_model.py \
  --dataset training/data/processed/temporal_sequences.jsonl \
  --output-dir training/models/temporal_lstm_v1 \
  --epochs 20
```

## 7. Hệ thống bắt cheat như thế nào sau khi thêm training

### 7.1 Tầng detector

Tầng này phát hiện:

- điện thoại
- tài liệu / giấy ghi chú
- người thứ hai

Phần này dùng `YOLO fine-tuned`.

### 7.2 Tầng temporal behavior

Tầng này không nhìn 1 frame đơn lẻ mà nhìn chuỗi frame theo thời gian.

Feature đầu vào lấy từ:

- số mặt trong frame
- phone confidence
- book confidence
- yaw ratio
- pitch ratio
- eye line angle
- event count
- các cờ như `face_missing`, `look_away`, `looking_down`

`LSTM` sẽ học sự khác nhau giữa:

- chớp mắt ngắn
- quay đầu rất ngắn
- cúi xuống ngắn
- hành vi gian lận kéo dài hoặc lặp lại

## 8. Gợi ý để dữ liệu đủ tốt

- Không quay toàn bộ một nhóm người theo cùng một phòng và cùng một ánh sáng.
- Không chỉ quay case gian lận, phải quay nhiều case bình thường ngắn.
- Không dùng mỗi ảnh tĩnh, nên có clip ngắn 3 đến 10 giây.
- Nên cân bằng số lượng clip giữa các nhãn.
- Nên giữ riêng `train`, `val`, `test` theo người, tránh để cùng một người xuất hiện ở cả train và test.

## 9. Kết luận

Sau khi bổ sung các thành phần trên, project không còn chỉ là demo heuristic. Nó đã có đường pipeline rõ ràng để:

- thu thập dữ liệu thật
- chuẩn hóa dữ liệu cho fine-tune
- train object detector
- train temporal behavior model

Đây là phần rất quan trọng để đáp ứng yêu cầu học thuật về `có huấn luyện AI` trong đề tài.
