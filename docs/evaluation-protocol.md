# Giao thức đánh giá

Tài liệu này cố định cách đo trước khi chạy thực nghiệm. Nhóm không đổi chỉ số hay ngưỡng sau khi đã thấy kết quả.

## 1. Chỉ số

| Chỉ số | Định nghĩa |
|---|---|
| Tỉ lệ thành công nhiệm vụ | Tỉ lệ lượt mà vật được chỉ định nằm trong khay được chỉ định khi kết thúc lượt, và không vật nào khác bị đặt sai khay |
| Tỉ lệ chọn đúng vật | Tỉ lệ lượt robot gắp đúng vật được chỉ định, kể cả khi đặt sai |
| Tỉ lệ gắp thành công | Tỉ lệ lượt robot nâng được vật lên khỏi bàn |
| Thời gian một lượt | Từ lúc nhận lệnh tới lúc robot thả vật |
| Số lần lớp an toàn chặn | Số lệnh bị lớp an toàn từ chối trong một lượt |
| Pha thất bại | Pha mà lượt thất bại: tiếp cận, gắp, nâng, di chuyển, đặt |

Trong mô phỏng, chương trình đo các chỉ số tự động từ trạng thái simulator. Trên robot thật, một thành viên quan sát và ghi vào phiếu đánh giá, có video kèm theo.

## 2. Thiết kế thí nghiệm

| Thí nghiệm | Câu hỏi | Điều kiện so sánh |
|---|---|---|
| Đánh giá trong mô phỏng | Mô hình đã tinh chỉnh đạt tỉ lệ thành công bao nhiêu? | Lệnh tiếng Việt và tiếng Anh; câu lệnh đã thấy và câu lệnh do người viết (chưa thấy) |
| So sánh với mô hình đối chứng | Thông tin ngôn ngữ giúp được bao nhiêu? | Mô hình VLA và ACT trên cùng tập cảnh |
| Đánh giá trên robot thật | Giữ được bao nhiêu hiệu năng khi chuyển sang robot thật? | Chỉ huấn luyện mô phỏng; chỉ huấn luyện demo thật; huấn luyện kết hợp |

Mở rộng nếu còn thời gian: phân loại theo nhóm ngữ nghĩa, và độ bền khi thay đổi vị trí camera, ánh sáng, vật gây nhiễu.

## 3. Số lần thử và thống kê

- Mỗi điều kiện chạy ít nhất 30 lượt.
- Báo cáo tỉ lệ thành công kèm khoảng tin cậy Wilson 95%.
- Các điều kiện cần so sánh với nhau dùng chung tập cảnh và seed.
- Báo cáo đủ số lượt thất bại. Không loại lượt nào sau khi đã xem kết quả.

Khoảng tin cậy Wilson cho $k$ lượt thành công trên $n$ lượt, với $\hat p = k/n$ và $z = 1{,}96$:

$$\frac{\hat p + \frac{z^2}{2n} \pm z\sqrt{\frac{\hat p(1-\hat p)}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}$$

## 4. Truy vết

Mỗi bảng kết quả trong báo cáo ghi:

- commit của repo và phiên bản dataset;
- checkpoint mô hình;
- danh sách seed cảnh;
- đường dẫn tới log đánh giá gốc.

Script trong repo sinh ra hình và bảng từ log đánh giá. Không chép số liệu bằng tay.
