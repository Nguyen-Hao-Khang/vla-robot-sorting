# Giai đoạn 1: pipeline mô-đun (tháng 9–10/2026)

Thư mục này lưu mã nguồn, cấu hình và bằng chứng thực nghiệm của giai đoạn đầu dự án. Nhóm không phát triển tiếp phần này. Mã được giữ để tham khảo và để các test cũ vẫn chạy trong CI.

## Nội dung

Giai đoạn 1 xây một hệ thống phân loại không dùng mô hình VLA, gồm các bước nối tiếp:

1. tri giác: tìm vật trong ảnh RGB-D từ camera mô phỏng;
2. hiểu lệnh: dùng mô hình ngôn ngữ chuyển câu lệnh thành vật đích và khay đích;
3. chọn tư thế gắp;
4. lập kế hoạch chuyển động bằng CuRobo và thực thi trong RoboTwin;
5. lớp an toàn và cơ chế phục hồi khi thao tác lỗi.

Thư viện dùng chung của các bước này nằm trong `src/vla_robot_sorting/` và vẫn được dùng ở giai đoạn hiện tại, ví dụ lớp an toàn và giao diện mô phỏng.

## Cấu trúc

| Thư mục | Nội dung |
|---|---|
| `scripts/` | Script chạy thí nghiệm, phân tích và kiểm tra bằng chứng |
| `configs/` | Cấu hình của các đợt thí nghiệm, gồm danh sách phiên bản phụ thuộc (`s1.1/repos.lock.yaml`) |
| `experiments/` | Kịch bản, tập câu lệnh, kết quả đã chấp nhận và báo cáo kiểm tra |
| `docs/` | Tài liệu kiến trúc và ADR của giai đoạn 1 |
| `tests/` | Test cho các script trên |

## Quan hệ với giai đoạn hiện tại

- Chuyên gia sinh demo trong `tools/twin/` kế thừa phần lập kế hoạch chuyển động và chọn tư thế gắp của giai đoạn 1.
- Lý do chuyển sang mô hình VLA đầu-cuối: [ADR 0002](../../docs/adr/0002-vla-dau-cuoi.md).
