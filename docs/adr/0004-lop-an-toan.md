# ADR 0004: Lớp an toàn bắt buộc giữa mô hình và robot thật

- Trạng thái: Chấp nhận
- Ngày: 2026-10-27

## Bối cảnh

Mô hình VLA có thể xuất hành động bất kỳ, kể cả vượt giới hạn khớp, ra ngoài bàn hoặc di chuyển quá nhanh. Panthera-HT có tải 3,5 kg và sải tay 860 mm, đủ để gây hư hỏng hoặc chấn thương.

## Quyết định

Mọi hành động gửi xuống robot thật phải đi qua lớp an toàn (`src/vla_robot_sorting/safety/`). Lớp này:

1. từ chối vị trí khớp ngoài giới hạn của nhà sản xuất;
2. từ chối vị trí đầu kẹp ra ngoài hộp vùng làm việc khai báo trong cấu hình;
3. giới hạn vận tốc khớp;
4. dừng robot khi mất tín hiệu từ policy quá thời gian chờ;
5. luôn cho phép dừng khẩn cấp bằng nút cứng.

Lớp an toàn cũng chạy trong mô phỏng để số lần chặn được đo giống nhau ở hai môi trường.

## Hệ quả

- Không ai được nới ngưỡng an toàn để một lần chạy qua được (CONTRIBUTING mục 5).
- Mỗi thay đổi ở lớp an toàn cần test và review của một thành viên khác.
- Số lần lớp an toàn chặn là một chỉ số đánh giá (`docs/giao-thuc-danh-gia.md`).
