# ADR 0005: Chọn mô hình VLA nền

- Trạng thái: Đang xem xét
- Hạn quyết định: xem `docs/project-plan.md`, giai đoạn 1

## Bối cảnh

Ràng buộc:

- suy luận được trên GPU 6 GB (GTX 1660 Ti, không có bf16);
- tinh chỉnh được trên RTX 5090 hoặc Kaggle T4 trong vài giờ;
- hiểu lệnh tiếng Việt;
- có sẵn trong LeRobot để dùng chung định dạng dữ liệu và công cụ huấn luyện.

## Phương án

| Mô hình | Kích thước | Mô hình ngôn ngữ bên trong | Ghi chú |
|---|---|---|---|
| SmolVLA | ~0,45 tỉ tham số | SmolLM2, huấn luyện chủ yếu trên tiếng Anh | Tiền huấn luyện trên dữ liệu cộng đồng LeRobot (cánh tay SO-100) |
| Evo-1 | ~0,77 tỉ tham số | Qwen2.5, đa ngôn ngữ | Có trong LeRobot |

ACT được dùng làm mô hình đối chứng trong mọi trường hợp, không phải ứng viên cho vai trò mô hình chính.

## Tiêu chí đo

Nhóm đo cả hai mô hình trên cùng một dataset nhỏ (khoảng 50 demo mô phỏng):

1. VRAM và độ trễ suy luận trên GTX 1660 Ti ở fp16;
2. thời gian tinh chỉnh 10.000 bước trên RTX 5090;
3. tỉ lệ thành công trong mô phỏng với 30 lượt lệnh tiếng Anh và 30 lượt lệnh tiếng Việt.

## Quyết định

Chưa có. Nhóm cập nhật mục này kèm bảng số đo khi hoàn thành phép so sánh.
