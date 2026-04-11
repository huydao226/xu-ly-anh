# Training Experiment Note

## 1. Mục đích của file này

File này ghi lại:

- dataset dùng để train model `oep_webcam_monitor_v3`
- cách chia `train / val / test`
- quy trình train
- tham số train chính
- các metric đánh giá
- ý nghĩa của từng metric
- kết quả hiện tại và các lưu ý quan trọng

## 2. Model đang được train

Model đang dùng cho OEP service là:

- `training/models/oep_webcam_monitor_v3`

Loại model:

- `LSTM` cho temporal classification

Các lớp hiện tại:

- `normal`
- `suspicious_action`
- `device`

## 3. Dataset dùng để train

Dataset đầu vào của lần train này là:

- [oep_webcam_temporal_v3.jsonl](/Users/huy.dao/XuLyAnh/anti-cheat-demo/training/data/processed/oep_webcam_temporal_v3.jsonl)

Nguồn dữ liệu:

- OEP dataset của Michigan State University
- chỉ dùng nhánh `webcam-only` cho monitor v3
- có thêm các sample `normal` được suy ra từ các khoảng trống không có cheating event

Tổng số sample:

- `827`

Số lớp:

- `3`

Số feature mỗi frame:

- `19`

### 3.1 Vì sao chỉ có 24 subject nhưng lại có hàng trăm sample

Điểm này rất quan trọng để hiểu đúng dataset.

Trong OEP:

- `subject` là người tham gia
- mỗi `subject` có video webcam dài
- trong video đó có nhiều đoạn hành vi khác nhau
- mỗi đoạn hành vi trong `gt.txt` được gọi là một `segment`

Ví dụ một file `gt.txt` có thể có nhiều dòng như:

```text
0135  0204  1
0205  0220  1
0221  0223  2
...
```

Mỗi dòng như vậy nghĩa là:

- từ thời điểm bắt đầu đến thời điểm kết thúc
- người đó thực hiện một hành vi có nhãn nhất định

Do đó:

- `24 subject` không có nghĩa là chỉ có `24 sample`
- một `subject` có thể tạo ra nhiều `segment`
- mỗi `segment` sau khi convert thành chuỗi feature sẽ trở thành `1 sample train`

Trong lần build hiện tại:

- parse từ `gt.txt` ra được `565` labeled segments
- suy thêm `262` normal segments từ các khoảng trống giữa các event
- tổng cộng thành `827` samples

Nói ngắn gọn:

- `subject` = người
- `segment` = đoạn hành vi trong video của người đó
- `sample` = một segment đã được convert thành sequence dùng để train

### 3.2 Vì sao vẫn phải cẩn thận khi đánh giá

Mặc dù có `827` sample, nhưng số người thật vẫn chỉ là `24`.

Điều đó có nghĩa là:

- số lượng sample tăng lên do cắt nhỏ theo thời gian
- nhưng độ đa dạng về người tham gia không tăng tương ứng

Vì vậy, việc chia `train / val / test` cần làm theo `subject`, không chia ngẫu nhiên theo sample, để tránh việc cùng một người xuất hiện ở cả train và test.

## 4. Cách chia dataset

Dataset được gán sẵn trường `split` trong file `.jsonl` và train script đọc trực tiếp trường này.

Kết quả chia tập hiện tại:

- `train`: `518` sample
- `val`: `167` sample
- `test`: `142` sample

### 4.1 Phân bố lớp theo từng split

#### Train

- `suspicious_action`: `346`
- `normal`: `169`
- `device`: `3`

#### Validation

- `suspicious_action`: `117`
- `normal`: `50`
- `device`: `0`

#### Test

- `suspicious_action`: `87`
- `normal`: `43`
- `device`: `12`

### 4.2 Nhận xét về cách chia tập

Điểm cần lưu ý:

- lớp `device` rất ít
- tập `validation` hiện tại không có sample `device`
- vì vậy các metric validation chưa phản ánh tốt khả năng tổng quát hóa của lớp `device`

Đây là một hạn chế thực tế của lần train hiện tại, cần nêu rõ trong báo cáo.

## 5. Quy trình train

Quy trình tổng quát:

1. Đọc toàn bộ sample từ file `.jsonl`
2. Tách `train`, `val`, `test` theo trường `split`
3. Tính `feature_mean` và `feature_std` trên tập `train`
4. Chuẩn hóa feature của mọi sequence theo thống kê từ tập train
5. Tính `class_weights` để giảm ảnh hưởng mất cân bằng lớp
6. Dùng `WeightedRandomSampler` khi train
7. Train `LSTM` theo từng epoch
8. Ở mỗi epoch:
   - train trên `train_loader`
   - đánh giá trên `val_loader`
9. Lưu `best_model.pt` theo `best_val_accuracy`
10. Sau khi train xong:
   - tune `non_normal_threshold` trên tập validation
   - evaluate trên tập test
   - lưu `metrics.json`
   - lưu `label_map.json`

## 6. Lệnh train thực tế

```bash
python training/scripts/train_temporal_model.py \
  --dataset training/data/processed/oep_webcam_temporal_v3.jsonl \
  --output-dir training/models/oep_webcam_monitor_v3 \
  --epochs 24 \
  --batch-size 16 \
  --balanced-sampler
```

## 7. Tham số train chính

Các tham số chính của lần train này:

- `epochs = 24`
- `batch_size = 16`
- `hidden_size = 64`
- `learning_rate = 0.0007`
- `weight_decay = 0.0001`
- `seed = 42`
- `max_frames = 90`
- `balanced_sampler = true`

### 7.1 Giải thích các tham số

#### `epochs`

- số vòng lặp train qua toàn bộ tập train

#### `batch_size`

- số sample được đưa vào model trong một lần update gradient

#### `hidden_size`

- số chiều của hidden state trong `LSTM`

#### `learning_rate`

- tốc độ cập nhật trọng số của optimizer

#### `weight_decay`

- regularization để giảm overfitting

#### `max_frames`

- số frame tối đa mà train script cho phép trong một sequence
- trong dataset v3 hiện tại, sequence thực tế ngắn hơn
- script vẫn pad/clamp được vì nó hỗ trợ chung cho nhiều dataset

#### `balanced_sampler`

- bật `WeightedRandomSampler`
- giúp model nhìn thấy lớp hiếm nhiều hơn trong quá trình train

## 8. Kiến trúc và cách đo

### 8.1 Model

Kiến trúc chính:

- `LSTM(input_size=19, hidden_size=64, batch_first=True)`
- head phân loại:
  - `Linear(64 -> 64)`
  - `ReLU`
  - `Dropout(0.2)`
  - `Linear(64 -> 3)`

### 8.2 Loss

- `CrossEntropyLoss`
- có dùng `class_weights`

### 8.3 Optimizer

- `Adam`

## 9. Class weights

Class weights hiện tại:

- `device`: `7.5865`
- `normal`: `1.0108`
- `suspicious_action`: `0.7064`

Ý nghĩa:

- lớp càng hiếm thì trọng số càng cao
- do `device` có rất ít sample trong train nên trọng số bị đẩy lên mạnh

## 10. Threshold được tune sau train

Sau khi train xong, hệ thống không dùng trực tiếp prediction top-1 cho mọi trường hợp.

Nó tune thêm:

- `non_normal_threshold = 0.75`

Ý nghĩa:

- nếu model dự đoán một lớp khác `normal`
- nhưng confidence thấp hơn `0.75`
- thì ép kết quả về `normal`

Mục tiêu:

- giảm false positive của các lớp bất thường

## 11. Các metric đang lưu

Các metric chính hiện có trong `metrics.json`:

- `best_val_accuracy`
- `val_accuracy_thresholded`
- `val_macro_f1_thresholded`
- `test_loss`
- `test_accuracy`
- `test_accuracy_thresholded`
- `test_macro_f1_thresholded`
- `val_prediction_counts_raw`
- `val_prediction_counts_thresholded`
- `test_prediction_counts_raw`
- `test_prediction_counts_thresholded`
- `history`

## 12. Ý nghĩa của từng metric

### `best_val_accuracy`

- accuracy tốt nhất trên tập validation trong toàn bộ quá trình train
- dùng để chọn checkpoint tốt nhất

### `val_accuracy_thresholded`

- accuracy trên tập validation sau khi áp dụng `non_normal_threshold`
- phản ánh cách model sẽ được dùng gần runtime hơn

### `val_macro_f1_thresholded`

- macro F1 trên validation sau threshold
- đo cân bằng giữa precision và recall ở từng lớp
- phù hợp hơn accuracy khi dataset bị lệch lớp

### `test_loss`

- loss trên tập test
- cho biết mức sai số tổng quát của model trên dữ liệu chưa dùng để train

### `test_accuracy`

- accuracy thô trên tập test trước khi áp dụng threshold

### `test_accuracy_thresholded`

- accuracy trên tập test sau khi áp dụng threshold
- đây là metric gần cách runtime sử dụng hơn

### `test_macro_f1_thresholded`

- macro F1 trên tập test sau threshold
- metric quan trọng khi muốn biết model có đang bỏ quên lớp hiếm hay không

### `prediction_counts_raw`

- số lần model dự đoán từng lớp trước threshold

### `prediction_counts_thresholded`

- số lần model dự đoán từng lớp sau threshold
- giúp thấy threshold làm dịch chuyển output như thế nào

### `history`

- log từng epoch:
  - `train_loss`
  - `train_accuracy`
  - `val_loss`
  - `val_accuracy`

## 13. Kết quả train hiện tại

Kết quả chính:

- `best_val_accuracy = 0.7066`
- `val_accuracy_thresholded = 0.6826`
- `val_macro_f1_thresholded = 0.4283`
- `test_loss = 2.5293`
- `test_accuracy = 0.5845`
- `test_accuracy_thresholded = 0.5211`
- `test_macro_f1_thresholded = 0.3651`
- `non_normal_threshold = 0.75`

### 13.1 Diễn giải nhanh

#### Validation accuracy cao hơn test accuracy

- cho thấy model có học được pattern trên validation
- nhưng khả năng tổng quát hóa ra test vẫn còn hạn chế

#### Macro F1 khá thấp

- đặc biệt quan trọng vì dataset bị lệch lớp mạnh
- nghĩa là model chưa cân bằng tốt giữa các lớp

#### Threshold làm accuracy test giảm

- `test_accuracy` giảm từ `58.45%` xuống `52.11%` sau threshold
- nhưng threshold vẫn có ý nghĩa practical vì nó giúp giảm các dự đoán non-normal yếu

## 14. Kết quả prediction counts

### Validation raw

- `device`: `1`
- `normal`: `40`
- `suspicious_action`: `126`

### Validation thresholded

- `device`: `0`
- `normal`: `61`
- `suspicious_action`: `106`

### Test raw

- `device`: `4`
- `normal`: `62`
- `suspicious_action`: `76`

### Test thresholded

- `device`: `3`
- `normal`: `81`
- `suspicious_action`: `58`

### Nhận xét

- threshold đẩy thêm nhiều sample về `normal`
- điều này làm giảm bớt non-normal prediction yếu
- nhưng cũng có thể làm giảm recall của lớp bất thường

## 15. Ý nghĩa thực tế cho hệ thống demo

Từ kết quả train này, hệ thống hiện đang áp dụng theo hướng practical:

- `Phase A`
  - temporal model chịu trách nhiệm chính cho `normal` và `suspicious_action`
- `Phase B`
  - `device` được hỗ trợ thêm bằng YOLO runtime override
- `Phase C`
  - `absence/offscreen` được xử lý bằng rule riêng

Lý do:

- lớp `device` trong dataset train còn quá ít
- validation không có sample `device`
- vì vậy nếu chỉ phụ thuộc vào temporal model thì demo sẽ kém ổn định

## 16. Hạn chế của lần train hiện tại

- dữ liệu `device` quá ít trong tập train
- tập `validation` không có sample `device`
- OEP không hoàn toàn khớp với điều kiện triển khai `1 camera laptop` thực tế
- một số segment `normal` vẫn có thể chứa đoạn cuối bị mất `face` hoặc `upper-body`
- macro F1 còn thấp, cho thấy model chưa cân bằng tốt giữa các lớp

## 17. Hướng cải thiện tiếp theo

- tăng số sample `device`
- chia lại split để `val` có đủ cả `3` lớp
- thêm dữ liệu `1 camera` riêng của đề tài
- giữ OEP như pretraining/reference dataset, sau đó fine-tune trên dữ liệu thật
- cân nhắc thay feature thủ công bằng backbone mạnh hơn nếu cần
