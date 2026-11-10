# Dữ liệu và huấn luyện

## 1. Nhiệm vụ

Mỗi demo là một lần gắp–đặt: robot gắp một vật trên bàn và đặt vào một khay theo câu lệnh. Đặc tả nhiệm vụ nằm trong `docs/dac-ta-nhiem-vu.md`, gồm:

- danh sách vật, màu và hình dạng;
- số khay và vị trí khay;
- cách sinh câu lệnh;
- tiêu chí thành công đo tự động trong mô phỏng.

## 2. Nguồn dữ liệu

| Nguồn | Cách tạo | Số lượng mục tiêu |
|---|---|---|
| Demo mô phỏng | Chuyên gia mô phỏng thực hiện thao tác trên cảnh ngẫu nhiên, `tools/twin/twin_record.py` ghi lại | 500–1.000 lượt thành công |
| Câu lệnh huấn luyện | Mô hình ngôn ngữ sinh nhiều cách diễn đạt tiếng Việt và tiếng Anh cho mỗi cặp (vật, khay) | ≥ 30 câu cho mỗi cặp |
| Câu lệnh kiểm tra | Người thật viết qua biểu mẫu, không dùng khi huấn luyện | 100–200 câu |
| Demo thật | Điều khiển tay chủ–tay tớ trên Panthera-HT | 30–50 lượt |

### 2.1 Ngẫu nhiên hóa cảnh

Một mô hình VLA thấy cùng một bố cục cảnh với cùng một câu lệnh sẽ học thuộc vị trí thay vì đọc câu lệnh. Bộ sinh dữ liệu phải đảm bảo:

- vị trí, màu và số lượng vật thay đổi giữa các bố cục;
- một bố cục đi với nhiều câu lệnh khác nhau, mỗi câu chỉ tới một vật khác;
- seed ngẫu nhiên được ghi lại để tái tạo chính xác bộ dữ liệu.

### 2.2 Câu lệnh

Mỗi câu lệnh sinh tự động lưu kèm:

- tên và phiên bản mô hình ngôn ngữ đã dùng;
- prompt;
- ngày sinh.

Nhóm loại bỏ các câu trùng lặp và các câu chỉ sai vật. Tập câu kiểm tra do người viết phải giữ tách biệt khỏi dữ liệu huấn luyện.

## 3. Định dạng

### 3.1 Dữ liệu thô (`twin_record.py`)

```
<out>/episode_000000/meta.json           câu lệnh, cấp độ cảnh, vật, khay, thành công, fps
<out>/episode_000000/states.npy          float32 [T, D]  vị trí khớp + kẹp
<out>/episode_000000/actions.npy         float32 [T, D]  vị trí khớp mục tiêu ở bước kế tiếp
<out>/episode_000000/<camera>/000000.jpg ảnh 320×240
```

Tần số ghi mặc định là 15 Hz (đặt trong `configs/twin_cameras.json`).

Lệnh ghi demo (chạy trong môi trường `RoboTwin`, không mở cửa sổ):

```bash
python tools/twin/twin_record.py --out data/twin_raw --episodes 50 --levels 1,2,3 --seed 0
```

### 3.2 Dataset LeRobot (`lerobot_convert.py`)

```bash
python tools/twin/lerobot_convert.py --raw data/twin_raw --root data/panthera_twin_v1 \
    --repo-id <tài khoản>/panthera_twin_v1
```

Đặt tên dataset theo phiên bản, ví dụ `panthera_twin_v1`, `panthera_twin_v2`. Không sửa một phiên bản sau khi đã dùng nó để huấn luyện mô hình được báo cáo.

## 4. Huấn luyện

| Mô hình | Vai trò | Nơi huấn luyện |
|---|---|---|
| Mô hình VLA đã chọn (SmolVLA hoặc Evo-1) | Mô hình chính | RTX 5090 hoặc Kaggle |
| ACT (Action Chunking with Transformers) | Mô hình đối chứng không dùng ngôn ngữ | Bất kỳ GPU ≥ 6 GB |

Mỗi lần huấn luyện được báo cáo phải lưu:

- commit của repo;
- phiên bản dataset;
- file cấu hình đầy đủ: siêu tham số, seed, số bước;
- checkpoint cuối và log huấn luyện, tải lên Hugging Face Hub hoặc Kaggle.

GPU đời Turing (GTX 1660 Ti, Kaggle T4) không hỗ trợ bf16 bằng phần cứng. Trên các GPU này, đặt kiểu dữ liệu là fp16 hoặc fp32.

## 5. Chạy policy trong mô phỏng

```bash
# môi trường LeRobot
python tools/twin/policy_bridge.py serve --checkpoint <đường dẫn checkpoint>

# môi trường RoboTwin, cửa sổ khác
python tools/twin/twin_console.py --executor policy --policy-server 127.0.0.1:5555 --level 2
```

Cách đánh giá và báo cáo kết quả: [giao-thuc-danh-gia.md](giao-thuc-danh-gia.md).
