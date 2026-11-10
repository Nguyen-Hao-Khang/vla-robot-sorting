# ADR 0001: Dùng RoboTwin với mô hình Panthera-HT làm môi trường mô phỏng

- Trạng thái: Chấp nhận
- Ngày: 2026-10-05

## Bối cảnh

Nhóm cần sinh hàng trăm demo gắp–đặt và đánh giá mô hình hàng trăm lượt trước khi có robot thật. Môi trường mô phỏng phải có mô hình đúng của cánh tay Panthera-HT, camera kết xuất được ảnh màu và chạy được trên laptop GTX 1660 Ti.

## Phương án

| Phương án | Ưu điểm | Nhược điểm |
|---|---|---|
| RoboTwin (SAPIEN, PhysX), bản Panthera-HT của hãng HighTorque | Hãng cung cấp sẵn mô hình robot; có sẵn bộ lập kế hoạch CuRobo; cộng đồng dùng để sinh dữ liệu VLA | Chỉ chạy trên Linux có GPU NVIDIA; tài liệu ít |
| Isaac Sim / Isaac Lab | Kết xuất đẹp, mô phỏng vật lý mạnh | Cần GPU RTX và nhiều VRAM; phải tự dựng mô hình Panthera-HT |
| MuJoCo | Nhẹ, nhanh | Phải tự chuyển mô hình robot; kết xuất ảnh kém thực tế hơn |

## Quyết định

Dùng Panthera-HT RoboTwin, cố định commit `2aee31ad`, CuRobo `v0.7.8`, SAPIEN `3.0.0b1` (xem `docs/getting-started.md`).

## Hệ quả

- Mô phỏng chỉ chạy trên máy Ubuntu có GPU NVIDIA. Các thành viên khác làm phần dữ liệu, huấn luyện và báo cáo trên Windows hoặc GPU đám mây.
- Code dự án truy cập RoboTwin qua `tools/twin/`, không sửa mã nguồn RoboTwin.
- Các giới hạn tầm với và vật cản của môi trường được ghi trong `docs/architecture.md` mục 5.
