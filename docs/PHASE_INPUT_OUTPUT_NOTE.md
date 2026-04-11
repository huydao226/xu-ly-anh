# Phase Input Output Note

## 1. Tổng quan

Runtime hiện tại của OEP monitor v3 có thể chia thành `3 work stream` chính:

- `Phase A`: Predict abnormal by `OEP v3 temporal model`
- `Phase B`: Check device by `YOLO`
- `Phase C`: Check absence by `rule`

Ba luồng này chạy trên cùng một frame/session context, sau đó được hợp nhất ở bước `merge decision`.

## 2. Input chung của hệ thống

Input gốc từ frontend là:

- `frame`: ảnh webcam hiện tại ở dạng `base64`
- `session_id`: id của phiên monitor

Ví dụ request:

```json
{
  "frame": "data:image/jpeg;base64,...",
  "captured_at": "optional"
}
```

Trước khi vào 3 phase, backend thực hiện bước chuẩn bị chung:

1. decode ảnh từ base64
2. resize frame về width chuẩn `320`
3. chuyển sang grayscale
4. lấy `previous_gray` từ session trước đó

Từ đây, dữ liệu được dùng cho 3 phase song song.

---

## 3. Phase A - Predict Abnormal by OEP v3 Model

### 3.1 Input

Input của phase này là:

- frame hiện tại sau preprocess
- `previous_gray`
- `feature_buffer` của session

Trong đó `feature_buffer` là `deque(maxlen=16)` chứa chuỗi vector đặc trưng của các frame gần nhất.

### 3.2 Process

Phase này chạy theo các bước:

1. gọi `extract_frame_features(...)`
2. build vector `19 features` cho frame hiện tại
3. append vector vào `feature_buffer`
4. nếu buffer có dưới `8` frame:
   - chưa predict
   - chỉ trả trạng thái `collecting temporal sequence`
5. nếu buffer có từ `8` frame trở lên:
   - lấy toàn bộ sequence hiện có
   - normalize bằng `feature_mean` và `feature_std`
   - đưa qua `LSTM`
   - lấy `softmax`
   - chọn lớp top-1
   - nếu lớp khác `normal` nhưng confidence thấp hơn `non_normal_threshold = 0.75` thì ép về `normal`

### 3.3 Output

Output của phase này là:

- `prediction_label`
  - `normal`
  - `suspicious_action`
  - `device`
- `confidence`
- `probabilities`
- `features` của frame hiện tại

### 3.4 Ý nghĩa

Đây là luồng AI chính để phân tích hành vi theo thời gian.

Nó không chỉ nhìn một ảnh đơn, mà nhìn một chuỗi frame đã được buffer lại.

---

## 4. Phase B - Check Device by YOLO

### 4.1 Input

Input của phase này là:

- frame hiện tại sau preprocess ảnh màu

Phase này không dùng temporal buffer.

### 4.2 Process

Phase này chạy độc lập với temporal model:

1. chạy `YOLO('yolov8n.pt')` trên frame
2. duyệt qua các detection box
3. chỉ giữ các class:
   - `cell phone`
   - `phone`
   - `laptop`
4. chọn box có confidence cao nhất
5. nếu confidence >= `0.7`
   - set `device_detected = true`
   - lưu `bbox`
   - set `device_hold_remaining = 12`
   - lưu `last_device_confidence`
6. nếu frame hiện tại không thấy device:
   - giảm `device_hold_remaining` theo từng frame
   - nếu hold vẫn còn thì `device_active` vẫn là `true`

### 4.3 Output

Output của phase này là:

- `device_detected`
- `device_active`
- `device_confidence`
- `device_bbox`

### 4.4 Ý nghĩa

Phase này là luồng practical để đảm bảo khi thấy điện thoại rõ thì nhãn `device` đủ ổn định cho demo, không bị nhảy mất quá nhanh.

---

## 5. Phase C - Check Absence by Rule

### 5.1 Input

Input của phase này là:

- `face_present`
- `pose_present`
- `offscreen_streak`

Các giá trị này đến từ kết quả của `extract_frame_features(...)` và session state.

### 5.2 Process

Phase này chạy như sau:

1. nếu `face_present = false` và `pose_present = false`
   - tăng `offscreen_streak`
2. nếu một trong hai xuất hiện lại
   - reset `offscreen_streak = 0`
3. nếu `offscreen_streak >= 6`
   - tạo nhãn override `absence/offscreen`
   - confidence tăng dần theo số frame vắng mặt

### 5.3 Output

Output của phase này là:

- `absence_active`
- `absence_confidence`
- `override_label = absence/offscreen` nếu đủ điều kiện

### 5.4 Ý nghĩa

Phase này giúp bắt trường hợp người dùng rời khỏi vùng quan sát, thay vì chỉ dựa vào temporal model.

---

## 6. Merge Decision

Sau khi 3 phase chạy xong, hệ thống mới đưa ra quyết định cuối.

### 6.1 Input của bước merge

Bước merge nhận:

- output từ `Phase A`
  - `prediction_label`
  - `confidence`
  - `probabilities`
- output từ `Phase B`
  - `device_active`
  - `device_confidence`
  - `device_bbox`
- output từ `Phase C`
  - `absence_active`
  - `absence_confidence`

### 6.2 Thứ tự merge

Thứ tự ưu tiên hiện tại là:

1. `device`
2. `absence/offscreen`
3. kết quả của `OEP v3 model`

### 6.3 Logic merge

#### Case 1: Device active

Nếu `device_active = true`:

- override nhãn cuối thành `device`
- confidence cuối lấy từ:
  - `device_confidence`
  - hoặc `last_device_confidence`
- status text: `Device rule triggered (...)`

Khi đó:

- bỏ qua `absence/offscreen`
- không dùng trực tiếp label của temporal model

#### Case 2: Không có device, nhưng absence active

Nếu `device_active = false` và `absence_active = true`:

- override nhãn cuối thành `absence/offscreen`
- confidence lấy từ rule
- status text báo số frame mất tín hiệu

Khi đó:

- label từ temporal model bị thay thế

#### Case 3: Không có device, không absence

Nếu cả hai rule trên không kích hoạt:

- giữ nguyên kết quả từ temporal model
- status text: `OEP monitor v3 predicts ...`

### 6.4 Output cuối của merge

Output cuối cùng trả về frontend gồm:

- `prediction_label`
- `confidence`
- `probabilities`
- `annotated_frame`
- `status_text`
- `features`
- `session summary`

---

## 7. Bảng tóm tắt Input / Process / Output

| Phase | Input | Process | Output |
|---|---|---|---|
| Phase A - OEP v3 model | frame, previous_gray, feature_buffer | extract 19 features -> append buffer -> normalize -> LSTM -> softmax | prediction_label, confidence, probabilities |
| Phase B - YOLO | current color frame | detect phone/laptop -> threshold -> hold frames | device_detected, device_active, device_confidence, bbox |
| Phase C - Absence rule | face_present, pose_present, offscreen_streak | streak counting -> threshold check | absence_active, absence_confidence |
| Merge | outputs from A, B, C | priority-based override | final decision for UI |

---

## 8. Cách giải thích ngắn khi thuyết trình

Có thể nói ngắn gọn như sau:

> Hệ thống hiện tại có ba luồng xử lý song song. Luồng thứ nhất là temporal model OEP v3 để dự đoán hành vi theo chuỗi frame. Luồng thứ hai là YOLO để bắt thiết bị như điện thoại hoặc laptop. Luồng thứ ba là rule để kiểm tra vắng mặt khi mất cả mặt và upper-body liên tiếp. Sau đó backend hợp nhất ba kết quả này theo thứ tự ưu tiên để tạo ra nhãn cuối cùng đưa lên giao diện.
