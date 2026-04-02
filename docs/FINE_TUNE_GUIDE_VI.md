# Hướng Dẫn Fine-Tune

## 1. Mục tiêu

Tài liệu này mô tả cách cung cấp dữ liệu và huấn luyện thêm AI cho đề tài giám sát gian lận thi trực tuyến bằng 1 camera laptop.

Sau khi cập nhật pipeline, project có thể đi theo hướng nghiên cứu thực tế hơn:

- `Stage 1`: train baseline trên `OEP multi-view`
- `Stage 2`: train hoặc fine-tune tiếp trên `OEP webcam-only`
- `Stage 3`: test trên `OEP webcam-only holdout`
- `Stage 4`: self-test với webcam laptop của nhóm

## 2. Dữ liệu cần chuẩn bị

### 2.1 Dataset chính

Ở giai đoạn hiện tại, dataset chính để train ngay có thể là `OEP`.

OEP phù hợp để làm:

- baseline research
- webcam-only adaptation
- holdout testing

Sau đó, nếu cần làm sát triển khai thật hơn, mới bổ sung dataset `1 camera` của chính đề tài như một giai đoạn sau.

Sau khi giải nén, OEP có cấu trúc:

- `24 subject`
- mỗi subject có `gt.txt`
- `1` file audio `.wav`
- `1` webcam video `.avi`
- `1` wearcam video `.avi`

Điểm này rất hữu ích vì cho phép bạn đi theo lập luận:

- train trên dữ liệu nhiều ngữ cảnh trước
- sau đó giới hạn lại chỉ còn webcam phía trước
- cuối cùng đánh giá trên holdout split của webcam-only

Điểm này rất quan trọng với các tình huống như:

- cúi đầu nhưng đang gõ bàn phím bình thường
- chỉnh ghế hoặc đổi tư thế ngắn
- chớp mắt khi đeo kính
- người có mắt nhỏ hoặc detector mắt không ổn định

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

## 3. Pipeline huấn luyện theo OEP trước

### 3.1 Stage 1: OEP multi-view

Mục tiêu:

- dùng cả `webcam` và `wearcam`
- học baseline hành vi từ dataset proctoring thật

Lệnh:

```bash
cd /Users/huy.dao/XuLyAnh/anti-cheat-demo/backend
source .venv/bin/activate
cd /Users/huy.dao/XuLyAnh/anti-cheat-demo

python training/scripts/import_oep_reference.py

python training/scripts/build_oep_temporal_dataset.py \
  --mode multiview \
  --output training/data/processed/oep_multiview_temporal.jsonl

python training/scripts/train_temporal_model.py \
  --dataset training/data/processed/oep_multiview_temporal.jsonl \
  --output-dir training/models/oep_multiview_lstm_v1 \
  --epochs 12
```

### 3.2 Stage 2: OEP webcam-only

Mục tiêu:

- chỉ giữ camera mặt
- mô phỏng gần hơn với điều kiện `1 camera laptop`

Lệnh:

```bash
cd /Users/huy.dao/XuLyAnh/anti-cheat-demo/backend
source .venv/bin/activate
cd /Users/huy.dao/XuLyAnh/anti-cheat-demo

python training/scripts/build_oep_temporal_dataset.py \
  --mode webcam \
  --output training/data/processed/oep_webcam_temporal.jsonl

python training/scripts/train_temporal_model.py \
  --dataset training/data/processed/oep_webcam_temporal.jsonl \
  --output-dir training/models/oep_webcam_lstm_v1 \
  --epochs 12
```

### 3.3 Stage 3: OEP webcam-only holdout test

Script builder tự chia `train`, `val`, `test` theo `subject`, không chia ngẫu nhiên theo clip. Điều này giúp tránh rò rỉ dữ liệu giữa các clip của cùng một người.

Khi chạy `train_temporal_model.py`, file `metrics.json` trong thư mục model sẽ chứa:

- `best_val_accuracy`
- `test_accuracy`

Đây chính là kết quả đánh giá chính thức đầu tiên cho pipeline OEP.

### 3.4 Stage 4: Self-test sau

Sau khi có kết quả trên OEP, bạn có thể dùng demo webcam hiện tại để kiểm tra định tính:

- model có phản ứng đúng khi quay đầu hay không
- các case cúi đầu ngắn có còn bị bắt nhầm hay không
- kết quả demo có hợp lý với người dùng thật hay không

## 4. Cách lấy dữ liệu ngay từ hệ thống hiện tại

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

## 4.1 Bạn phải đặt dataset ở đâu để tôi chạy được ngay

Sau khi bạn tải dataset, hãy đặt vào đúng các thư mục sau:

```text
/Users/huy.dao/XuLyAnh/anti-cheat-demo/training/data/external/oep_multiview/raw
/Users/huy.dao/XuLyAnh/anti-cheat-demo/training/data/external/single_camera_finetune/raw_videos
/Users/huy.dao/XuLyAnh/anti-cheat-demo/training/data/external/single_camera_finetune/raw_images
/Users/huy.dao/XuLyAnh/anti-cheat-demo/training/data/external/single_camera_finetune/annotations
```

Ý nghĩa hiện tại:

- `oep_multiview/raw`
  - chứa OEP hoặc dataset tham chiếu public tương tự
- `single_camera_finetune/*`
  - chưa bắt buộc ngay lúc này
  - để dành cho giai đoạn sau nếu muốn thích nghi sát hơn với laptop webcam thật của nhóm

Sau khi tải xong, tôi có thể kiểm tra nhanh bằng:

```bash
cd /Users/huy.dao/XuLyAnh/anti-cheat-demo
python3 training/scripts/check_dataset_ready.py
```

Nếu bạn đã giải nén OEP, tôi có thể parse luôn manifest tham chiếu bằng:

```bash
cd /Users/huy.dao/XuLyAnh/anti-cheat-demo
python3 training/scripts/import_oep_reference.py
```

Script này sẽ tạo:

- `training/data/external/oep_multiview/notes/oep_subject_manifest.csv`
- `training/data/external/oep_multiview/notes/oep_webcam_segments.csv`
- `training/data/external/oep_multiview/notes/oep_summary.json`

Mục đích của các file này là:

- xem nhanh OEP có những subject nào
- lấy riêng webcam stream để đối chiếu với bài toán của đề tài
- chọn các đoạn cần relabel hoặc trích frame
- không dùng trực tiếp thay cho dataset fine-tune 1 camera

## 5. Bạn cần cung cấp dữ liệu fine-tune như thế nào

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

Lưu ý cho giai đoạn sau:

- OEP có `gt.txt` nhưng đó là nhãn theo dataset gốc
- để phù hợp với hệ thống hiện tại, bạn vẫn nên relabel lại theo schema của project
- đặc biệt với các case `normal but suspicious-looking` như cúi đầu do đánh máy, chỉnh ghế hoặc quay đầu ngắn

## 6. Cách build dataset

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

## 7. Cách train

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
