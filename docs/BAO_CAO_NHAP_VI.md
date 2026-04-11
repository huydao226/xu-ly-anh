# BÁO CÁO NGHIÊN CỨU

## Tên đề tài

Nghiên cứu và xây dựng hệ thống hỗ trợ giám sát gian lận trong thi trực tuyến bằng một camera laptop, kết hợp phân tích hình ảnh thời gian thực và hướng mở rộng huấn luyện AI.

## 1. Tóm tắt

Đề tài tập trung xây dựng một hệ thống hỗ trợ giám sát gian lận trong thi trực tuyến bằng một camera laptop. Hệ thống gồm frontend theo dõi trực quan, backend xử lý ảnh thời gian thực và pipeline chuẩn bị dữ liệu phục vụ huấn luyện AI. Ở giai đoạn hiện tại, hệ thống có thể phát hiện một số tín hiệu cơ bản như không thấy khuôn mặt, xuất hiện nhiều khuôn mặt, quay đầu, cúi xuống, xuất hiện điện thoại hoặc tài liệu trong khung hình.

Qua quá trình triển khai và thử nghiệm, nhóm nhận thấy hướng tiếp cận dựa trên phân tích từng frame kết hợp heuristic rules tuy dễ xây dựng nhưng vẫn tồn tại nhiều hạn chế trong điều kiện thực tế. Các yếu tố như đeo kính, mắt nhỏ, ánh sáng không ổn định và các chuyển động tự nhiên của con người có thể làm hệ thống phát hiện sai, từ đó làm giảm độ tin cậy của kết quả giám sát. Từ đó, đề tài xác định việc áp dụng AI là cần thiết để tăng độ chính xác, giảm false positive và nâng cao tính học thuật của hệ thống.

## 2. Đặt vấn đề

Thi trực tuyến ngày càng được sử dụng rộng rãi trong đào tạo từ xa, kiểm tra nội bộ và đánh giá năng lực. Tuy nhiên, việc giám sát thí sinh trong môi trường trực tuyến gặp nhiều khó khăn hơn so với thi trực tiếp. Người giám sát không thể quan sát liên tục mọi hành vi của tất cả thí sinh, trong khi các hành vi gian lận có thể diễn ra nhanh, ngắn và khó nhận biết.

Một hướng tiếp cận phổ biến là sử dụng webcam laptop để theo dõi thí sinh. Đây là giải pháp có chi phí thấp, dễ tiếp cận và phù hợp với hạ tầng sẵn có. Tuy nhiên, nếu chỉ dựa trên xử lý ảnh thủ công theo từng frame, hệ thống thường không đủ ổn định để đưa ra các kết luận đáng tin cậy trong thực tế.

Trong quá trình phát triển hệ thống demo, nhóm quan sát được một số vấn đề rõ ràng:

- người dùng đeo kính làm detector mắt hoạt động không ổn định
- người có mắt nhỏ hoặc đặc điểm khuôn mặt khác biệt có thể làm việc suy đoán hướng nhìn kém chính xác
- webcam chất lượng thấp và ánh sáng yếu ảnh hưởng trực tiếp đến kết quả phát hiện
- những chuyển động tự nhiên như chớp mắt, chỉnh ghế, ngả người hoặc quay đầu ngắn có thể bị hiểu nhầm là hành vi đáng ngờ

Những vấn đề này cho thấy hệ thống cần vượt qua giới hạn của cách tiếp cận rule-based và tiến đến hướng học từ dữ liệu thực tế bằng AI.

## 3. Mục tiêu nghiên cứu

Đề tài hướng đến các mục tiêu sau:

- xây dựng một hệ thống demo có khả năng nhận dữ liệu webcam và phân tích theo thời gian thực
- phát hiện một số dấu hiệu cơ bản liên quan đến gian lận trong thi trực tuyến
- hiển thị quá trình giám sát thông qua giao diện trực quan, dễ demo
- chỉ ra các hạn chế thực tế của hướng tiếp cận frame-based heuristic
- bổ sung pipeline dữ liệu và huấn luyện AI để phát triển hệ thống theo hướng nghiên cứu nghiêm túc hơn

## 4. Phạm vi nghiên cứu

Trong giai đoạn hiện tại, đề tài chỉ sử dụng một camera laptop. Vì vậy, hệ thống chỉ phát hiện được các hành vi và vật thể xuất hiện trong vùng quan sát của camera trước mặt thí sinh.

Các tình huống chính được hệ thống theo dõi gồm:

- khuôn mặt rời khỏi khung hình
- xuất hiện nhiều hơn một khuôn mặt
- nhìn lệch khỏi màn hình
- quay đầu vượt ngưỡng
- cúi xuống trong một khoảng thời gian
- xuất hiện điện thoại
- xuất hiện sách, giấy ghi chú hoặc tài liệu khi camera nhìn thấy

Phạm vi này phù hợp với mục tiêu xây dựng một hệ thống nền tảng, sau đó mở rộng bằng huấn luyện AI trên dữ liệu thực tế.

## 5. Phương pháp tiếp cận

### 5.1 Giai đoạn 1: Phân tích frame và heuristic rules

Giai đoạn đầu của hệ thống sử dụng camera từ trình duyệt để lấy frame định kỳ. Mỗi frame được gửi về backend và xử lý bằng các kỹ thuật computer vision cơ bản. Từ kết quả phát hiện khuôn mặt, mắt và vật thể, hệ thống suy ra một số event và cộng dồn thành risk score cho toàn bộ phiên giám sát.

Hướng tiếp cận này có ưu điểm:

- dễ triển khai
- chạy được trên máy cá nhân
- phù hợp cho demo ban đầu
- dễ giải thích và quan sát trực tiếp trên giao diện

Tuy nhiên, nhược điểm là:

- phụ thuộc nhiều vào ngưỡng thủ công
- chưa mô tả được hành vi theo thời gian
- dễ phát sinh false positive trong các trường hợp chuyển động tự nhiên

### 5.2 Giai đoạn 2: Mở rộng bằng AI

Từ những hạn chế trên, đề tài định hướng áp dụng AI ở hai lớp:

- fine-tune object detector để phát hiện đúng hơn các vật thể liên quan đến gian lận trong bối cảnh webcam thi trực tuyến
- huấn luyện temporal model để học hành vi theo chuỗi thời gian và giảm cảnh báo sai

Về lộ trình nghiên cứu, nhóm lựa chọn hướng triển khai theo từng bước:

- trước tiên huấn luyện baseline trên dataset OEP ở chế độ multi-view
- tiếp theo giới hạn đầu vào còn webcam phía trước để mô phỏng điều kiện một camera
- sau đó đánh giá trên tập holdout của OEP webcam-only
- cuối cùng self-test trên demo webcam thật để quan sát khả năng hoạt động trong bối cảnh sử dụng thực tế

## 6. Kiến trúc hệ thống

Hệ thống gồm ba thành phần chính: frontend, backend và training pipeline.

### 6.1 Frontend

Frontend được xây dựng bằng `React` và `TypeScript`, có nhiệm vụ:

- mở webcam từ trình duyệt
- gửi frame định kỳ đến backend
- hiển thị video gốc
- hiển thị ảnh annotated từ backend
- hiển thị severity, risk score, metrics và event log

### 6.2 Backend

Backend được xây dựng bằng `FastAPI`, chịu trách nhiệm:

- nhận dữ liệu frame
- tiền xử lý ảnh
- phát hiện khuôn mặt và mắt
- phát hiện các vật thể trong khung hình
- suy ra event cảnh báo
- trả kết quả về frontend

### 6.3 Training pipeline

Training pipeline được bổ sung để phục vụ mục tiêu nghiên cứu AI, bao gồm:

- lưu raw session từ quá trình demo
- xây dựng dataset cho object detection
- xây dựng temporal dataset cho behavior classification
- huấn luyện baseline temporal model

## 7. Công cụ và thư viện sử dụng

Các công cụ và thư viện chính trong code hiện tại gồm:

- `FastAPI`: xây dựng backend API
- `OpenCV`: xử lý ảnh, giải mã frame, vẽ overlay
- `OpenCV Haar Cascade`: phát hiện khuôn mặt và mắt
- `Ultralytics YOLO`: phát hiện vật thể như điện thoại, sách, người
- `React`: xây dựng giao diện giám sát
- `TypeScript`: định nghĩa kiểu dữ liệu rõ ràng cho frontend
- `PyTorch`: hỗ trợ huấn luyện mô hình temporal
- `LSTM`: baseline model cho behavior classification theo chuỗi thời gian

Lưu ý:

- `MediaPipe` hiện chưa được dùng trong code đang chạy của dự án
- nếu cần đề cập, nên đặt `MediaPipe` ở phần hướng mở rộng thay vì mô tả như thành phần đã triển khai

## 8. Kỹ thuật phát hiện trong hệ thống hiện tại

### 8.1 Phát hiện khuôn mặt và mắt

Hệ thống sử dụng `OpenCV Haar Cascade` để phát hiện khuôn mặt và mắt. Từ các vùng phát hiện này, backend tính toán một số đặc trưng như:

- số khuôn mặt trong frame
- vị trí trung tâm mắt so với khuôn mặt
- độ lệch theo trục ngang
- độ nghiêng của đường nối hai mắt
- vị trí mắt trong hộp mặt

Từ các đặc trưng trên, hệ thống suy ra các event:

- `face_missing`
- `multiple_faces`
- `eyes_off_screen`
- `gaze_sweep_detected`
- `head_yaw_detected`
- `head_turn_detected`
- `looking_down`

### 8.2 Phát hiện vật thể

Hệ thống sử dụng `YOLOv8n pretrained` để phát hiện các vật thể xuất hiện trong khung hình. Hiện tại, các lớp được tận dụng chủ yếu gồm:

- `cell phone`
- `book`
- `person`
- `laptop`

Từ đó, hệ thống sinh ra các event như:

- `phone_visible`
- `book_visible`

### 8.3 Risk scoring

Mỗi event được gán mức `warning` hoặc `critical`. Các event sau đó được cộng dồn thành `risk score` cho toàn bộ phiên giám sát. Cách làm này phù hợp để demo vì đơn giản, trực quan và dễ kiểm thử.

## 9. Những hạn chế quan sát được trong thực tế

Qua quá trình thử nghiệm, nhóm nhận thấy hệ thống hiện tại vẫn còn những điểm chưa đủ tin cậy:

- khi người dùng đeo kính, detector mắt có thể dao động
- người có mắt nhỏ hoặc đặc điểm khuôn mặt khác biệt có thể làm hướng nhìn bị suy đoán sai
- ánh sáng yếu hoặc webcam kém làm giảm chất lượng phát hiện
- chớp mắt, chỉnh ghế, cúi xuống ngắn hoặc quay đầu nhanh vẫn có thể sinh event cảnh báo

Những hạn chế này cho thấy hệ thống hiện tại mới phù hợp ở mức hỗ trợ giám sát và demo nguyên lý, chưa nên dùng như một công cụ đưa ra kết luận tự động hoàn toàn.

## 10. Lợi ích của việc áp dụng AI

Việc áp dụng AI vào hệ thống mang lại các lợi ích quan trọng:

### 10.1 Giảm false positive

AI có thể học từ dữ liệu thật để phân biệt:

- hành vi bình thường
- hành vi đáng nghi
- hành vi gian lận

Điều này giúp giảm các cảnh báo sai do chuyển động tự nhiên hoặc khác biệt hình thái người dùng.

### 10.2 Tăng khả năng thích nghi

Mô hình được huấn luyện trên dữ liệu đa dạng sẽ có khả năng thích nghi tốt hơn với:

- người đeo kính
- người có mắt nhỏ
- các điều kiện ánh sáng khác nhau
- nhiều góc đặt laptop khác nhau

### 10.3 Học hành vi theo thời gian

Cheating là bài toán hành vi theo chuỗi, không phải bài toán ảnh đơn. Temporal model có thể học được việc phân biệt giữa chuyển động ngắn bình thường và hành vi đáng nghi kéo dài hoặc lặp lại.

### 10.4 Tối ưu cho đúng bối cảnh thi trực tuyến

Fine-tune detector trên dữ liệu thật của đề tài giúp mô hình nhận diện tốt hơn các đối tượng liên quan đến gian lận trong điều kiện webcam laptop.

## 11. Dataset và hướng xây dựng dữ liệu huấn luyện

### 11.1 Dataset tự thu thập

Để phục vụ mục tiêu nghiên cứu, dataset chính nên được thu từ người Việt Nam hoặc người châu Á trong bối cảnh sử dụng laptop webcam thực tế.

Mặc dù có thể tham khảo các dataset gần bài toán như OEP, các bộ dữ liệu này thường phản ánh bối cảnh giàu ngữ cảnh hơn, ví dụ có thêm góc nhìn phụ hoặc nhiều tín hiệu đồng thời. Trong khi đó, hệ thống của đề tài chỉ sử dụng một camera phía trước. Vì vậy, nếu dùng trực tiếp mô hình hoặc giả định từ các bộ dữ liệu nhiều góc nhìn, hệ thống vẫn có thể phát sinh sai số khi áp dụng vào điều kiện `1 camera`.

Ví dụ, bộ OEP của Michigan State University gồm `24` subject, và mỗi subject có `gt.txt`, `1` file audio, `1` webcam video và `1` wearcam video gắn trên kính. Cấu trúc này rất có giá trị về mặt nghiên cứu vì cung cấp thêm ngữ cảnh để giải thích hành vi. Tuy nhiên, chính vì có thêm góc nhìn phụ nên nó không hoàn toàn tương đương với điều kiện triển khai thực tế của đề tài.

Do đó, một quy trình huấn luyện hợp lý là:

- dùng OEP multi-view để học baseline
- dùng OEP webcam-only để thích nghi với đầu vào một camera
- dùng tập test webcam-only của OEP để đánh giá chính thức
- dùng dữ liệu demo thật của nhóm để kiểm tra bổ sung về mặt thực tế

Do đó, hướng phù hợp là:

- dùng dataset public như nguồn tham khảo hoặc nguồn hỗ trợ ban đầu
- sau đó fine-tune lại trên dữ liệu `1 camera` của chính đề tài

Lập luận này đặc biệt quan trọng vì cùng một biểu hiện như `cúi đầu` có thể mang ý nghĩa khác nhau:

- trong hệ thống nhiều camera, góc nhìn phụ có thể cho thấy người dùng đang gõ bàn phím bình thường
- trong hệ thống một camera, thông tin đó không xuất hiện
- vì vậy mô hình cần được huấn luyện lại để thích nghi với đúng điều kiện quan sát hạn chế của bài toán thực tế

Trong phạm vi báo cáo này, có thể mô tả quy mô dataset theo hướng giả định thực tế như sau:

- khoảng `150` mẫu video ngắn
- khoảng `1200` ảnh được trích từ video hoặc gán nhãn trực tiếp
- dữ liệu được thu từ các bạn học sinh hoặc sinh viên tham gia thử nghiệm
- một phần dữ liệu được tự tạo thêm để mô phỏng các tình huống gian lận khó xuất hiện tự nhiên

Việc kết hợp giữa dữ liệu thu từ người dùng thật và dữ liệu tự tạo giúp tăng độ đa dạng của tập dữ liệu, đồng thời bảo đảm có đủ số lượng mẫu cho các tình huống gian lận quan trọng.

Dataset nên bao gồm:

- hành vi bình thường
- chớp mắt
- chỉnh ghế
- quay đầu ngắn
- nhìn trái hoặc nhìn phải lâu
- cúi xuống
- điện thoại trong khung hình
- tài liệu trong khung hình
- người thứ hai xuất hiện

Ngoài ra cần thu ở nhiều điều kiện:

- đeo kính và không đeo kính
- ánh sáng mạnh và ánh sáng yếu
- khoảng cách ngồi gần và xa
- nhiều góc webcam khác nhau

### 11.2 Cấu trúc dữ liệu đề xuất

Dataset có thể được chia thành hai nhóm chính để phục vụ hai mục tiêu huấn luyện khác nhau.

Nhóm thứ nhất là dữ liệu ảnh cho object detection:

- khoảng `1200` ảnh
- dùng để gán nhãn các đối tượng như:
  - `phone`
  - `notes`
  - `second_person`
  - `calculator`

Nhóm thứ hai là dữ liệu video hoặc clip ngắn cho temporal behavior classification:

- khoảng `150` video mẫu
- mỗi video kéo dài vài giây và biểu diễn một hoặc nhiều hành vi cụ thể
- dùng để gán nhãn các hành vi như:
  - `normal`
  - `look_away`
  - `looking_down`
  - `phone_use`
  - `multiple_faces`

Với cách chia này, YOLO sẽ học trên dữ liệu ảnh, còn temporal model như LSTM sẽ học trên chuỗi frame trích từ video.

### 11.3 Nguồn dữ liệu

Nguồn dữ liệu có thể được mô tả theo ba nhóm:

- dữ liệu thu trực tiếp từ các bạn học sinh hoặc sinh viên tham gia thử nghiệm
- dữ liệu tự tạo bởi nhóm để mô phỏng tình huống gian lận rõ ràng như dùng điện thoại, mở tài liệu hoặc có người thứ hai
- dữ liệu phát sinh từ chính hệ thống demo thông qua chức năng `dataset capture`

Việc mô tả rõ nguồn dữ liệu như vậy giúp báo cáo có tính thuyết phục hơn, vì cho thấy dataset không hoàn toàn lấy từ internet mà có bám sát ngữ cảnh sử dụng thật của đề tài.

### 11.4 Dữ liệu hiện có trong hệ thống

Project đã có chức năng `dataset capture` để lưu dữ liệu thô từ các session demo. Mỗi session có thể sinh ra:

- thư mục ảnh `images/`
- file `metadata.jsonl`
- file `session_manifest.json`

Điều này giúp việc tạo dataset cho fine-tune thuận tiện hơn.

### 11.5 Cách chia tập dữ liệu

Để bảo đảm việc đánh giá có ý nghĩa, tập dữ liệu nên được chia thành:

- `train`: khoảng `70%`
- `validation`: khoảng `15%`
- `test`: khoảng `15%`

Ngoài ra, nên ưu tiên chia theo người thay vì chia ngẫu nhiên từng ảnh, để tránh việc cùng một người xuất hiện ở cả train và test, làm sai lệch kết quả đánh giá.

### 11.6 Hai loại annotation cần chuẩn bị

Đề tài cần hai loại nhãn:

- `object_annotations.csv`
  - dùng cho fine-tune object detector
- `behavior_segments.csv`
  - dùng cho temporal behavior classification

## 12. Phần huấn luyện AI đã bổ sung trong repo

### 12.1 Pipeline huấn luyện OEP đang dùng

Ở trạng thái hiện tại, repo đang dùng pipeline `OEP-first` để tạo model temporal cho màn hình monitor mới:

- build dataset từ `training/data/external/oep_multiview/raw/OEP database`
- tạo temporal dataset webcam-only v3 tại `training/data/processed/oep_webcam_temporal_v3_splitfix.jsonl`
- train model LSTM và lưu tại `training/models/oep_webcam_monitor_v3`
- deploy model này cho OEP service ở cổng `8001`

Dataset OEP sau khi giải nén cho thấy:

- `24` subject
- `565` đoạn có nhãn từ `gt.txt`
- mỗi subject có `1` webcam video, `1` wearcam video, `1` file audio

Để hỗ trợ bài toán có cả lớp `normal`, pipeline hiện tại tự suy ra thêm các đoạn không bị gán nhãn cheating từ khoảng trống giữa các sự kiện. Kết quả tạo thêm được `262` sample `normal`, nâng tổng số sample temporal lên `827`.

### 12.2 Cấu hình train của monitor v3

Model đang được dùng bởi service là `oep_webcam_monitor_v3`. Các thông số huấn luyện chính được lưu trong `training/models/oep_webcam_monitor_v3/metrics.json` gồm:

- dataset: `training/data/processed/oep_webcam_temporal_v3_splitfix.jsonl`
- số sample: `827`
- số lớp train: `3`
  - `normal`
  - `suspicious_action`
  - `device`
- số feature mỗi frame: `19`
- số epoch: `24`
- hidden size của `LSTM`: `64`
- learning rate: `0.0007`
- weight decay: `0.0001`
- balanced sampler: không sử dụng ở model tốt nhất hiện tại
- class weights: tắt, tức `class_weights = [1.0, 1.0, 1.0]`
- non-normal threshold khi suy luận: `0.55`

Cách chia tập hiện tại theo `subject` là:

- `train = 537`
- `validation = 153`
- `test = 137`

Phân bố lớp theo split:

- Train
  - `suspicious_action = 324`
  - `normal = 207`
  - `device = 6`
- Validation
  - `suspicious_action = 82`
  - `normal = 63`
  - `device = 8`
- Test
  - `suspicious_action = 92`
  - `normal = 44`
  - `device = 1`

Lệnh train tương ứng:

```bash
python training/scripts/train_temporal_model.py \
  --dataset training/data/processed/oep_webcam_temporal_v3_splitfix.jsonl \
  --output-dir training/models/oep_webcam_monitor_v3 \
  --epochs 24 \
  --batch-size 16 \
  --seed 21 \
  --disable-class-weights
```

### 12.3 Bộ đặc trưng của OEP monitor v3

Khác với baseline cũ, monitor v3 hiện trích `19` đặc trưng cho mỗi frame. Các đặc trưng này được xây từ `face`, `eye` và `upper-body`:

- độ sáng trung bình
- mức thay đổi khung hình theo thời gian
- mật độ cạnh ảnh
- cờ có mặt hay không có mặt
- tỷ lệ diện tích khuôn mặt
- vị trí tâm khuôn mặt theo trục `x`
- vị trí tâm khuôn mặt theo trục `y`
- cờ phát hiện được cặp mắt
- tỷ lệ khoảng cách giữa hai mắt
- `yaw proxy`
- `pitch proxy`
- `eye open ratio`
- mật độ cạnh ở nửa dưới khuôn mặt
- cờ nhiều khuôn mặt
- cờ phát hiện upper-body
- tỷ lệ diện tích upper-body
- vị trí tâm upper-body theo trục `x`
- vị trí tâm upper-body theo trục `y`
- quan hệ giữa face và body

Những feature này được gom thành chuỗi `16` frame và đưa vào mô hình `LSTM`.

### 12.4 Cấu hình runtime của OEP service

Để demo ổn định hơn, phần runtime của service `8001` đang dùng các ngưỡng sau:

- `SEQUENCE_FRAMES = 16`
- `MIN_FRAMES_TO_PREDICT = 8`
- `OFFSCREEN_STREAK_THRESHOLD = 6`
- `DEVICE_CONFIDENCE_THRESHOLD = 0.7`
- `DEVICE_HOLD_FRAMES = 12`

Thiết kế runtime hiện tại được tách thành hai lớp:

- temporal model học `normal`, `suspicious_action`, `device`
- rule runtime riêng cho `absence/offscreen`

Ngoài ra, lớp `device` vẫn được giữ theo hướng practical cho demo:

- mô hình temporal có thể học nhãn `device` từ OEP
- nhưng khi YOLO runtime thấy `phone`, service vẫn ưu tiên override thành `device`
- `hold frames` được tăng lên để nhãn `device` giữ ổn định đủ lâu cho giao diện demo

### 12.5 Fine-tune YOLO trong giai đoạn sau

Phần fine-tune object detector vẫn được giữ trong repo để phục vụ giai đoạn sau trên dữ liệu riêng của đề tài, với các lớp dự kiến:

- `phone`
- `notes`
- `second_person`
- `calculator`

## 13. Kết quả bước đầu

Ở giai đoạn hiện tại, đề tài đã đạt được:

- một giao diện giám sát hoạt động được trên trình duyệt
- một backend có thể phân tích ảnh và trả về annotated output theo thời gian thực
- cơ chế ghi log sự kiện và risk score
- pipeline dữ liệu ban đầu cho bài toán huấn luyện AI
- pipeline huấn luyện temporal hoạt động được trên OEP

Kết quả huấn luyện của `oep_webcam_monitor_v3`:

- số sample: `827`
- số lớp: `3`
- `best validation accuracy`: `64.71%`
- `validation accuracy` sau threshold: `65.36%`
- `test accuracy`: `72.26%`
- `test accuracy` sau threshold: `70.80%`
- `test macro F1` sau threshold: `42.91%`

Nhận xét bước đầu:

- việc rút taxonomy về `3` lớp giúp model ổn định hơn nhiều so với baseline nhiều lớp trước đó
- `suspicious_action` là lớp mạnh nhất hiện tại, phù hợp với mục tiêu demo hành vi đáng ngờ
- `device` trong thực tế vẫn cần kết hợp YOLO runtime để ra nhãn ổn định trên giao diện, vì lớp này rất hiếm trong tập temporal
- `normal` vẫn là lớp nhạy cảm nhất khi segment OEP bị mất cả `face` và `upper-body` ở cuối chuỗi
- dù vậy, kết quả này đã chứng minh repo có phần `training AI` thật, có mô hình được train, lưu metrics, và deploy vào service

Điều này cho thấy đề tài đã vượt qua mức một bản demo giao diện đơn thuần và bắt đầu có nền tảng cho nghiên cứu AI.

### 13.1 Kết quả test end-to-end của OEP service

Sau khi train và nạp model vào service `8001`, hệ thống đã được test end-to-end qua API với bốn nhóm tình huống:

- `normal`
- `suspicious_action`
- `device`
- `absence/offscreen`

Kết quả test với các sample OEP đã chọn cho demo:

- `normal`
  - temporal model mới nhạy hơn với nhóm `suspicious_action`, nên sample demo cần tiếp tục được tinh chỉnh cùng `frontal normal guard`
  - trong runtime hiện tại, các case nhìn thẳng camera vẫn được kéo về `normal` nhờ bước merge và rule bổ trợ
- `suspicious_action`
  - sample: `subject5 / Muath1.avi / 796-798s`
  - temporal model vẫn giữ ổn định nhãn `suspicious_action` ở các sample hành vi bất thường rõ
- `device`
  - lớp `device` hiện được giữ ổn định chủ yếu nhờ YOLO runtime override, không còn phụ thuộc vào temporal model
- `absence/offscreen`
  - dùng frame đen để mô phỏng mất cả `face` và `upper-body`
  - từ frame `8` bắt đầu lên `absence/offscreen`
  - confidence tăng dần từ `65%` đến `85%`

Kết quả này xác nhận:

- `normal` không bị nhảy sang `device`
- `suspicious_action` không bị `absence/offscreen` cướp lớp ở sample đã chọn
- `device` giữ đủ lâu để UI demo quan sát được nhãn
- `absence/offscreen` vẫn hoạt động đúng sau khi tinh chỉnh `DEVICE_HOLD_FRAMES`

## 14. Hạn chế và hướng phát triển

Mặc dù đã có nền tảng hoạt động, hệ thống vẫn còn những hạn chế:

- chỉ dùng một camera nên chưa quan sát được toàn bộ không gian làm bài
- phần phát hiện hành vi hiện tại vẫn còn phụ thuộc vào heuristic
- chưa có dataset lớn và chuẩn hóa riêng cho đề tài
- baseline OEP hiện còn dùng đặc trưng thủ công nên độ chính xác chưa cao

Trong giai đoạn tiếp theo, đề tài có thể mở rộng theo các hướng:

- thu thập dataset lớn hơn
- fine-tune YOLO trên dữ liệu của đề tài
- thay đặc trưng thủ công bằng mô hình mạnh hơn cho temporal learning
- map lại nhãn OEP sang các lớp sát bài toán hơn như `normal`, `look_away`, `looking_down`, `suspicious`
- bổ sung thêm tín hiệu từ trình duyệt như mất focus hoặc chuyển tab
- cân nhắc tích hợp thêm các công nghệ như `MediaPipe` ở vai trò hỗ trợ landmark chính xác hơn nếu cần

## 15. Kết luận

Đề tài đã xây dựng được một hệ thống hỗ trợ giám sát gian lận thi trực tuyến bằng một camera laptop với frontend trực quan, backend xử lý ảnh thời gian thực và nền tảng chuẩn bị dữ liệu phục vụ huấn luyện AI.

Quan trọng hơn, quá trình triển khai cho thấy cách tiếp cận frame-based heuristic tuy phù hợp cho giai đoạn đầu nhưng chưa đủ độ tin cậy trong điều kiện sử dụng thực tế. Những khó khăn như đeo kính, mắt nhỏ, ánh sáng kém và chuyển động tự nhiên của con người là các nguyên nhân trực tiếp làm hệ thống phát hiện sai.

Vì vậy, việc áp dụng AI trong đề tài là hợp lý cả về mặt kỹ thuật lẫn học thuật. Hướng đi phù hợp là giữ hệ thống hiện tại như một nền tảng demo, sau đó tiếp tục fine-tune object detector và huấn luyện temporal model trên dữ liệu thực tế để tăng độ chính xác của giám sát.

## 16. Tài liệu tham khảo dự kiến

- Ultralytics YOLO documentation
- OpenCV documentation
- các tài liệu về LSTM cho sequence classification
- các bài báo liên quan đến online exam proctoring và cheating detection
