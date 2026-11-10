# Đặc tả nhiệm vụ

Bản nháp. Nhóm chốt đặc tả này trước khi sinh dataset huấn luyện đầu tiên, và không đổi sau đó trừ khi tăng phiên bản dataset.

## 1. Cảnh

- Bàn làm việc trước robot, kích thước và vị trí khớp với bàn thật (đo khi robot về).
- 3 khay cố định: trái, giữa, phải.
- Mỗi cảnh có 2–4 vật đặt ngẫu nhiên trong vùng gắp, không chồng lên nhau.

## 2. Vật

| Thuộc tính | Giá trị |
|---|---|
| Hình dạng | khối lập phương, hình trụ, khối chữ nhật |
| Màu | đỏ, xanh lá, xanh dương, vàng |
| Kích thước | cạnh 4–6 cm, vừa kẹp hai ngón |

Hình tứ diện bị loại vì kẹp hai ngón gắp không ổn định (`docs/kien-truc-he-thong.md` mục 5).

## 3. Câu lệnh

Mỗi câu lệnh chỉ định một vật bằng màu, hình dạng hoặc cả hai, và một khay. Ví dụ:

- "gắp khối màu đỏ bỏ vào khay bên trái"
- "put the blue cylinder in the middle tray"
- "cho cái hình trụ vàng vào khay phải"

Câu lệnh phải xác định duy nhất một vật trong cảnh. Bộ sinh cảnh kiểm tra điều kiện này.

## 4. Tiêu chí thành công

Một lượt thành công khi, lúc kết thúc lượt:

1. vật được chỉ định nằm trong khay được chỉ định;
2. không vật nào khác bị dời vào khay nào;
3. lớp an toàn không dừng robot.

Thời gian tối đa một lượt: 60 giây.
