# Kiến trúc hệ thống

## 1. Bài toán

Đầu vào của hệ thống:

- Một câu lệnh tiếng Việt hoặc tiếng Anh, ví dụ "gắp khối màu đỏ bỏ vào khay bên trái".
- Ảnh từ hai camera: một camera gắn trên cổ tay robot và một camera tổng quan nhìn bàn làm việc.
- Trạng thái 6 khớp và độ mở kẹp của cánh tay Panthera-HT.

Đầu ra là chuỗi vị trí khớp mục tiêu. Hệ thống thực thi chuỗi này để gắp đúng vật và đặt vào đúng khay.

Phạm vi phân loại: theo **màu** và **hình dạng** của vật ([đặc tả nhiệm vụ](dac-ta-nhiem-vu.md)). Phân loại theo nhóm ngữ nghĩa (ví dụ "đồ nhựa", "đồ ăn được") là phần mở rộng nếu còn thời gian.

## 2. Thành phần

```
          câu lệnh VI/EN
                 │
┌────────────────▼─────────────────┐
│   Mô hình VLA đã tinh chỉnh      │◄── ảnh camera cổ tay + camera tổng quan
│   (chạy cục bộ, GPU ≥ 6 GB)      │◄── trạng thái khớp
└────────────────┬─────────────────┘
                 │ chuỗi vị trí khớp
┌────────────────▼─────────────────┐
│   Lớp an toàn                    │  giới hạn khớp, vùng làm việc,
│                                  │  vận tốc, dừng khẩn cấp
└────────────────┬─────────────────┘
                 │
     ┌───────────┴───────────┐
     ▼                       ▼
 Panthera-HT trong      Panthera-HT thật
 RoboTwin (mô phỏng)    (SDK của hãng)
```

| Thành phần | Vai trò | Vị trí trong repo |
|---|---|---|
| Mô hình VLA | Ánh xạ (ảnh, câu lệnh, trạng thái) → chuỗi hành động. Nhóm đang chọn giữa SmolVLA và Evo-1, xem [ADR 0005](adr/0005-lua-chon-mo-hinh-vla.md) | Huấn luyện bằng LeRobot; notebook trong `notebooks/` |
| Lớp an toàn | Chặn lệnh vượt giới hạn trước khi gửi xuống robot | `src/vla_robot_sorting/safety/` |
| Môi trường mô phỏng | Panthera-HT trong RoboTwin (SAPIEN, PhysX), cảnh bàn với vật và khay | `tools/twin/twin_scene.py`, `tools/twin/twin_console.py` |
| Chuyên gia mô phỏng | Thực hiện thao tác gắp–đặt bằng lập kế hoạch chuyển động (CuRobo) và thông tin ground truth của simulator, dùng để sinh demo | `tools/twin/twin_console.py`, `twin_grasp.py`, `twin_kinematics.py`, `twin_obstacles.py` |
| Ghi dữ liệu | Ghi trạng thái khớp, hành động và ảnh camera khi chuyên gia thao tác | `tools/twin/twin_record.py` |
| Chuyển đổi dataset | Chuyển dữ liệu thô sang định dạng LeRobot | `tools/twin/lerobot_convert.py` |
| Cầu nối policy | Chạy policy đã huấn luyện trong môi trường LeRobot và trả hành động cho mô phỏng qua socket cục bộ | `tools/twin/policy_bridge.py` |
| Thư viện dự án | Giao diện dữ liệu, adapter mô phỏng, lập kế hoạch, an toàn, đánh giá | `src/vla_robot_sorting/` |

## 3. Luồng dữ liệu

### 3.1 Huấn luyện

1. `twin_record.py` sinh cảnh ngẫu nhiên. Chuyên gia mô phỏng thực hiện một thao tác gắp–đặt, và hệ thống ghi dữ liệu ở 15 Hz:
   - trạng thái khớp;
   - vị trí khớp mục tiêu ở bước kế tiếp (dùng làm hành động);
   - ảnh camera cổ tay và camera tổng quan.
   Hệ thống chỉ giữ các lượt thành công.
2. Mô hình ngôn ngữ sinh nhiều cách diễn đạt câu lệnh tiếng Việt và tiếng Anh cho mỗi thao tác.
3. `lerobot_convert.py` chuyển dữ liệu thô sang dataset LeRobot. Nhóm lưu dataset trên Hugging Face Hub hoặc Kaggle.
4. Nhóm tinh chỉnh mô hình VLA trên GPU đám mây hoặc GPU ≥ 16 GB.
5. Khi robot thật sẵn sàng, nhóm thu thêm demo bằng điều khiển tay chủ–tay tớ (teleoperation) và tinh chỉnh tiếp trên dữ liệu kết hợp.

Chi tiết: [du-lieu-va-huan-luyen.md](du-lieu-va-huan-luyen.md).

### 3.2 Suy luận

1. `policy_bridge.py serve` nạp checkpoint và lắng nghe trên `127.0.0.1:5555`.
2. Mô phỏng hoặc chương trình điều khiển robot gửi trạng thái, ảnh và câu lệnh. Mỗi gói tin gồm 4 byte độ dài rồi tới nội dung JSON.
3. Policy trả về hành động. Lớp an toàn kiểm tra hành động trước khi thực thi.

Policy và mô phỏng chạy ở hai môi trường Python khác nhau (LeRobot và RoboTwin). Socket cục bộ tách hai môi trường đó, giống cách robot thật nhận lệnh từ máy tính điều khiển.

## 4. Ràng buộc phần cứng

| Máy | Cấu hình | Vai trò |
|---|---|---|
| Laptop mô phỏng | GTX 1660 Ti 6 GB, Ubuntu 22.04 | Chạy RoboTwin, sinh demo, đánh giá trong mô phỏng, điều khiển robot thật |
| Máy huấn luyện | RTX 5090, Windows | Tinh chỉnh mô hình VLA |
| GPU đám mây | Kaggle T4 16 GB, Colab | Tinh chỉnh khi máy huấn luyện bận |
| Robot | Panthera-HT, 6 khớp + kẹp, sải tay 860 mm, tải 3,5 kg | Thực nghiệm thật |

GTX 1660 Ti không hỗ trợ bf16 bằng phần cứng, nên mô hình phải suy luận ở fp16 hoặc fp32.

## 5. Giới hạn đã biết của môi trường mô phỏng

- **Tầm với khi kẹp hướng thẳng xuống.**
  - Kẹp của Panthera-HT chỉ lên được độ cao 0,86–0,92 m khi hướng thẳng xuống.
  - Khi nghiêng 35–45°, kẹp lên được 1,07–1,10 m.
  - Hệ quả: chuyên gia mô phỏng chọn tư thế gắp nghiêng khi cần (`twin_kinematics.py`).
- **Vật cản trong CuRobo.** Mặc định, RoboTwin chỉ đưa mặt bàn vào bản đồ vật cản của CuRobo. `twin_obstacles.py` bổ sung các vật trên bàn.
- **Vật khó gắp.** Kẹp hai ngón gắp vật hình tứ diện không ổn định. Đặc tả nhiệm vụ không dùng hình này.
