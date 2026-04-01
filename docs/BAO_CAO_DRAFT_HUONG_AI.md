# Bản Draft Báo Cáo Theo Hướng AI

## Tên đề tài

Nghiên cứu và xây dựng hệ thống hỗ trợ giám sát gian lận trong thi trực tuyến bằng một camera laptop, kết hợp phân tích hình ảnh thời gian thực và hướng mở rộng bằng AI để tăng độ chính xác.

## 1. Đặt vấn đề

Trong bối cảnh thi trực tuyến, việc giám sát thí sinh bằng camera laptop là một giải pháp phổ biến vì dễ triển khai và không yêu cầu thêm thiết bị chuyên dụng. Tuy nhiên, nếu chỉ dựa trên các luật xử lý ảnh đơn giản theo từng frame thì kết quả giám sát thường chưa ổn định và khó đạt độ tin cậy cao trong thực tế.

Qua quá trình xây dựng hệ thống demo, nhóm nhận thấy cách tiếp cận dựa trên `frame capture` và `heuristic rules` có thể phát sinh nhiều sai số. Một số trường hợp thường gặp gồm:

- người dùng đeo kính làm việc phát hiện mắt kém ổn định hơn
- người có mắt nhỏ hoặc mí mắt hẹp khiến việc suy ra hướng nhìn thiếu chính xác
- ánh sáng yếu hoặc webcam chất lượng thấp làm giảm độ tin cậy của detector
- con người không thể ngồi hoàn toàn bất động, nên các chuyển động tự nhiên như chớp mắt, chỉnh ghế, nghiêng đầu nhẹ hoặc cúi xuống ngắn có thể bị hiểu nhầm là hành vi đáng ngờ

Những hạn chế trên cho thấy nếu chỉ dùng rule-based detection thì hệ thống dễ sinh ra `false positive`, tức là cảnh báo sai đối với các hành vi bình thường. Đây chính là lý do cần áp dụng AI vào bài toán, nhằm giúp hệ thống học được mẫu hành vi thực tế thay vì chỉ phụ thuộc vào ngưỡng thủ công.

## 2. Mục tiêu nghiên cứu

Đề tài hướng tới các mục tiêu sau:

- xây dựng một hệ thống demo có thể nhận hình ảnh từ webcam laptop và phân tích theo thời gian thực
- phát hiện một số dấu hiệu liên quan đến gian lận như mất mặt, nhiều khuôn mặt, nhìn lệch, cúi xuống, điện thoại hoặc tài liệu trong khung hình
- hiển thị trực quan quá trình giám sát thông qua giao diện web
- làm rõ những hạn chế của cách tiếp cận frame-based heuristic trong điều kiện thực tế
- đề xuất và triển khai hướng mở rộng bằng AI để tăng độ chính xác và độ tin cậy của hệ thống

## 3. Hạn chế của hướng tiếp cận chỉ dựa trên frame capture

Hệ thống ở giai đoạn đầu sử dụng webcam từ trình duyệt, gửi từng frame về backend để phân tích. Cách làm này có ưu điểm là đơn giản, dễ demo và đủ để tạo ra một hệ thống mẫu hoạt động được. Tuy nhiên, phân tích độc lập từng frame không đủ để mô tả chính xác hành vi của con người theo thời gian.

Ví dụ:

- một lần chớp mắt ngắn không phải là hành vi gian lận
- một lần quay đầu rất nhanh rồi quay lại chưa đủ cơ sở để kết luận nhìn ra ngoài màn hình
- cúi xuống trong 1 đến 2 giây có thể chỉ là chỉnh tư thế ngồi
- khi đeo kính, detector mắt có thể dao động làm cho chỉ số hướng nhìn thay đổi bất thường

Như vậy, vấn đề cốt lõi không chỉ là phát hiện có gì trong ảnh, mà còn là phân biệt được đâu là chuyển động tự nhiên và đâu là hành vi đáng nghi kéo dài hoặc lặp lại. Đây là điểm mà AI, đặc biệt là mô hình học theo dữ liệu và theo chuỗi thời gian, mang lại lợi ích rõ rệt.

## 4. Lợi ích của việc áp dụng AI vào bài toán

Việc bổ sung AI vào hệ thống mang lại các lợi ích sau:

### 4.1 Giảm cảnh báo sai

Mô hình AI có thể học từ dữ liệu thực tế để phân biệt giữa:

- hành vi bình thường
- hành vi đáng nghi
- hành vi gian lận rõ ràng

Điều này giúp giảm số lượng cảnh báo sai do người dùng đeo kính, mắt nhỏ, thay đổi tư thế hoặc các cử động tự nhiên.

### 4.2 Tăng khả năng thích nghi với nhiều đối tượng

Nếu chỉ dùng luật cứng thì hệ thống thường hoạt động tốt trong một số điều kiện nhất định nhưng kém ổn định khi thay đổi người dùng, ánh sáng hoặc góc camera. AI có thể được huấn luyện trên dữ liệu đa dạng hơn, từ đó cải thiện khả năng thích nghi với nhiều nhóm người và nhiều bối cảnh khác nhau.

### 4.3 Học hành vi theo thời gian

Các hành vi gian lận thường không thể kết luận từ một ảnh đơn lẻ. Mô hình temporal như `LSTM` hoặc `Temporal CNN` có thể học từ chuỗi frame và đưa ra quyết định dựa trên diễn biến của hành vi theo thời gian, thay vì chỉ dựa vào một frame riêng lẻ.

### 4.4 Cải thiện detector vật thể cho đúng bối cảnh thi

Các model pretrained tổng quát có thể phát hiện điện thoại hoặc sách ở mức cơ bản, nhưng chưa tối ưu cho môi trường webcam thi trực tuyến. Fine-tune model trên dữ liệu thật của đề tài giúp cải thiện việc phát hiện:

- điện thoại
- giấy ghi chú
- tài liệu
- người thứ hai

## 5. Kiến trúc hệ thống hiện tại

Hệ thống hiện được chia thành ba phần chính:

### 5.1 Frontend

Frontend được xây dựng bằng `React` và `TypeScript`, có nhiệm vụ:

- truy cập webcam từ trình duyệt
- chụp frame định kỳ
- gửi frame đến backend
- hiển thị camera gốc, ảnh annotated, metric và log giám sát

### 5.2 Backend

Backend được xây dựng bằng `FastAPI`, có nhiệm vụ:

- tiếp nhận frame từ frontend
- xử lý ảnh
- phát hiện khuôn mặt và mắt
- phát hiện vật thể trong khung hình
- trả về event, severity, risk score và ảnh annotated

### 5.3 Training pipeline

Đây là phần được bổ sung để phục vụ mục tiêu nghiên cứu AI, bao gồm:

- lưu lại raw session từ quá trình demo
- chuẩn hóa dữ liệu cho object detection
- tạo temporal dataset từ chuỗi frame
- huấn luyện baseline temporal model

## 6. Công cụ và thư viện đang sử dụng trong code

Nếu bám đúng theo code hiện tại, các công cụ và thư viện chính đang được sử dụng gồm:

- `FastAPI`
  - xây dựng backend API
- `OpenCV`
  - xử lý ảnh, chuyển đổi frame, vẽ annotated output
- `OpenCV Haar Cascade`
  - phát hiện khuôn mặt và mắt
- `Ultralytics YOLO`
  - phát hiện vật thể như điện thoại, sách, người
- `React`
  - xây dựng giao diện người dùng
- `TypeScript`
  - tăng tính rõ ràng và an toàn kiểu dữ liệu cho frontend
- `PyTorch`
  - dùng trong phần training temporal model
- `LSTM`
  - baseline model để học hành vi theo chuỗi thời gian

Lưu ý quan trọng:

- `MediaPipe` hiện **không nằm trong code đang chạy** của project này
- nếu muốn nhắc `MediaPipe` trong báo cáo, nên đặt ở phần `hướng mở rộng` hoặc `công nghệ có thể tích hợp thêm`, không nên ghi là hệ thống hiện tại đã sử dụng

## 7. Hướng áp dụng AI trong đề tài

Để tăng độ chính xác cho hệ thống, đề tài định hướng áp dụng AI theo hai lớp.

### 7.1 Fine-tune object detector

Sử dụng `YOLO` và fine-tune trên dữ liệu webcam của chính đề tài để phát hiện tốt hơn các đối tượng liên quan đến gian lận:

- phone
- notes
- second_person
- calculator

### 7.2 Train temporal behavior model

Từ các chỉ số trích xuất theo thời gian như:

- số khuôn mặt
- phone detected
- yaw ratio
- pitch ratio
- eye line angle
- event flags

có thể huấn luyện mô hình `LSTM` để phân biệt:

- chuyển động tự nhiên ngắn
- hành vi đáng nghi
- hành vi gian lận

Đây là phần quan trọng nhất để giải quyết các trường hợp detector hiện tại còn bắt sai.

## 8. Dataset đề xuất

Để huấn luyện mô hình phù hợp với bối cảnh của đề tài, dataset nên được thu từ người Việt Nam hoặc người châu Á trong điều kiện sử dụng laptop webcam thực tế.

Những tình huống nên có trong dataset:

- hành vi bình thường
- chớp mắt
- chỉnh ghế
- quay đầu ngắn
- nhìn trái hoặc nhìn phải lâu
- cúi xuống
- dùng điện thoại
- xem tài liệu
- có người thứ hai

Ngoài ra, nên có dữ liệu trong các điều kiện:

- đeo kính và không đeo kính
- mắt nhỏ hoặc khuôn mặt có khác biệt hình thái
- ánh sáng mạnh và ánh sáng yếu
- khoảng cách ngồi gần và xa camera

## 9. Kết luận

Qua quá trình xây dựng hệ thống, có thể thấy rằng hướng tiếp cận dựa trên frame capture và heuristic là phù hợp để làm bản demo ban đầu, nhưng chưa đủ đáng tin cậy nếu triển khai như một hệ thống giám sát nghiêm túc trong thực tế.

Các yếu tố như đeo kính, mắt nhỏ, chất lượng webcam thấp và chuyển động tự nhiên của con người đều có thể làm hệ thống phát hiện sai. Vì vậy, việc áp dụng AI là hợp lý và cần thiết để nâng cao độ chính xác, giảm false positive và tăng tính học thuật cho đề tài.

Hướng đi phù hợp là giữ lại hệ thống demo hiện tại làm nền tảng, sau đó bổ sung fine-tune `YOLO` cho object detection và huấn luyện `LSTM` cho temporal behavior classification trên dữ liệu thực tế do nhóm thu thập.

## 10. Gợi ý cách trình bày trong báo cáo chính

Nếu dùng hướng này cho báo cáo chính, cấu trúc nên đi theo mạch:

1. Hệ thống frame-based ban đầu hoạt động như thế nào
2. Những lỗi thực tế quan sát được khi áp dụng cho người dùng thật
3. Vì sao các lỗi này làm giảm độ tin cậy của hệ thống
4. Vì sao cần AI
5. Hệ thống hiện tại đã bổ sung pipeline AI ra sao
6. Hướng huấn luyện và dataset cần có trong giai đoạn tiếp theo
