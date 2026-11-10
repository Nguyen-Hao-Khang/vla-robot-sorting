# ADR 0002: Mô hình VLA đầu-cuối là hệ thống chính

- Trạng thái: Chấp nhận
- Ngày: 2026-10-27

## Bối cảnh

Đề tài yêu cầu tích hợp công nghệ VLA với cánh tay robot để phân loại vật theo lệnh. Giai đoạn đầu, nhóm xây một pipeline mô-đun: tri giác → hiểu lệnh → chọn điểm gắp → lập kế hoạch chuyển động. Pipeline này hoạt động trong mô phỏng nhưng không dùng mô hình VLA.

## Phương án

| Phương án | Mô tả |
|---|---|
| A. VLA đầu-cuối | Một mô hình nhận ảnh, câu lệnh, trạng thái khớp và xuất trực tiếp chuỗi vị trí khớp |
| B. Mô hình thị giác–ngôn ngữ (VLM) lập kế hoạch + VLA thực thi | VLM tách lệnh thành bước con, VLA thực thi từng bước |
| C. Pipeline mô-đun | Tri giác, hiểu lệnh, lập kế hoạch chuyển động viết tay |

## Quyết định

Chọn A. Phương án B và C không nằm trong phạm vi học kỳ này.

Lý do:

- A đáp ứng trực tiếp yêu cầu đề tài và chạy cục bộ, không phụ thuộc dịch vụ trực tuyến khi vận hành.
- B thêm một mô hình lớn và một giao diện cần đánh giá riêng, không vừa thời gian còn lại.
- C không phải công nghệ VLA. Mã của C được giữ trong `legacy/sprint1/` và tái dùng làm chuyên gia sinh demo trong mô phỏng (ADR 0003).

## Hệ quả

- Chất lượng hệ thống phụ thuộc chủ yếu vào dữ liệu huấn luyện. Ngẫu nhiên hóa cảnh và đa dạng câu lệnh trở thành yêu cầu bắt buộc.
- Mô hình không đảm bảo an toàn, nên robot thật cần lớp an toàn riêng (ADR 0004).
