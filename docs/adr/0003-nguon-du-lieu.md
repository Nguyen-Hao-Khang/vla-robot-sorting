# ADR 0003: Sinh dữ liệu bằng chuyên gia mô phỏng và mô hình ngôn ngữ

- Trạng thái: Chấp nhận
- Ngày: 2026-10-27

## Bối cảnh

Tinh chỉnh một mô hình VLA cần hàng trăm demo có ảnh, trạng thái khớp, hành động và câu lệnh. Robot thật chỉ có từ cuối tháng 11, và thu demo bằng tay mất khoảng 1–2 phút mỗi lượt.

## Phương án

| Phương án | Ưu điểm | Nhược điểm |
|---|---|---|
| Chỉ thu demo thật | Không có khoảng cách mô phỏng–thực tế | Không đủ thời gian để thu hàng trăm lượt |
| Chuyên gia mô phỏng + mô hình ngôn ngữ sinh câu lệnh, bổ sung ít demo thật | Sinh được số lượng lớn, có nhãn tự động, tái tạo được bằng seed | Có khoảng cách mô phỏng–thực tế; chuyên gia dùng thông tin ground truth |
| Dùng dataset công khai (Open X-Embodiment, LIBERO) | Có sẵn | Khác robot, khác nhiệm vụ, không có tiếng Việt |

## Quyết định

Chọn phương án hai:

- Chuyên gia mô phỏng (lập kế hoạch chuyển động bằng CuRobo, đọc vị trí vật từ simulator) thực hiện thao tác, `twin_record.py` ghi lại.
- Mô hình ngôn ngữ lớn sinh nhiều cách diễn đạt tiếng Việt và tiếng Anh cho mỗi thao tác.
- Bổ sung 30–50 demo thật khi có robot.

## Hệ quả

- Thông tin ground truth chỉ dùng để sinh demo. Mô hình VLA chỉ nhận ảnh, câu lệnh và trạng thái khớp.
- Câu lệnh kiểm tra do người viết, tách khỏi câu lệnh huấn luyện, để đo khả năng hiểu lệnh chưa thấy.
- Chi tiết quy trình: `docs/du-lieu-va-huan-luyen.md`.
