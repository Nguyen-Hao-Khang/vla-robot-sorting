# Phân loại vật thể bằng cánh tay robot và mô hình Vision-Language-Action

Đồ án HK261, Khoa Khoa học và Kỹ thuật Máy tính, Trường Đại học Bách khoa – ĐHQG-HCM.
Giảng viên hướng dẫn: TS. Lê Trọng Nhân.
Nhóm thực hiện: Văn Quốc Huy, Nguyễn Hảo Khang, Phan Gia Phúc.

## Đề tài

Người vận hành ra lệnh bằng tiếng Việt hoặc tiếng Anh, ví dụ "gắp khối màu đỏ bỏ vào khay bên trái". Cánh tay robot Panthera-HT (6 bậc tự do, kẹp hai ngón) quan sát bàn làm việc qua camera rồi gắp và đặt vật vào đúng khay.

Nhóm giải bài toán bằng một mô hình Vision-Language-Action (VLA) nhỏ:

- **Đầu vào:** ảnh camera, câu lệnh và trạng thái khớp.
- **Đầu ra:** chuỗi lệnh khớp cho cánh tay.

Nhóm tinh chỉnh mô hình bằng dữ liệu sinh tự động trong mô phỏng, rồi bổ sung một lượng nhỏ dữ liệu thu trên robot thật.

## Mục tiêu

1. Tinh chỉnh một mô hình VLA nhỏ, chạy được trên GPU 6 GB khi suy luận, để phân loại vật theo màu và hình dạng từ lệnh tiếng Việt và tiếng Anh.
2. Xây dựng quy trình sinh dữ liệu huấn luyện trong mô phỏng RoboTwin: chuyên gia mô phỏng thực hiện thao tác, mô hình ngôn ngữ sinh câu lệnh đa dạng.
3. Tích hợp mô hình với cánh tay Panthera-HT thật qua một lớp an toàn giới hạn vùng làm việc, vận tốc và dừng khẩn cấp.
4. Đánh giá định lượng trong mô phỏng và trên robot thật: tỉ lệ thành công, độ hiểu lệnh, độ tin cậy thao tác.

## Trạng thái

Dự án đang ở giai đoạn chuẩn bị dữ liệu và chọn mô hình. Kế hoạch chi tiết nằm trong [docs/project-plan.md](docs/project-plan.md).

| Thành phần | Trạng thái |
|---|---|
| Mô phỏng Panthera-HT trong RoboTwin | Hoạt động |
| Chuyên gia mô phỏng ghi demo (`tools/twin`) | Hoạt động, cần mở rộng ngẫu nhiên hóa cảnh |
| Chuyển dữ liệu sang định dạng LeRobot | Hoạt động |
| Tinh chỉnh mô hình VLA | Đang chọn mô hình |
| Robot thật | Chờ nhận thiết bị (dự kiến tháng 11/2026) |

## Cấu trúc thư mục

| Thư mục | Nội dung |
|---|---|
| `src/vla_robot_sorting/` | Thư viện Python của dự án: giao diện dữ liệu, mô phỏng, tri giác, lập kế hoạch, an toàn, đánh giá |
| `tools/twin/` | Môi trường mô phỏng tương tác, chuyên gia sinh demo, ghi dữ liệu, chuyển đổi LeRobot, cầu nối chạy policy |
| `tools/setup/` | Script cài đặt và kiểm tra môi trường |
| `configs/` | Cấu hình camera và cảnh mô phỏng |
| `notebooks/` | Notebook huấn luyện trên Kaggle |
| `tests/` | Kiểm thử đơn vị, chạy không cần GPU hay mô phỏng |
| `report/` | Báo cáo khoa học (LaTeX, tiếng Việt) |
| `docs/` | Tài liệu kỹ thuật và kế hoạch dự án |
| `legacy/sprint1/` | Pipeline mô-đun giai đoạn đầu và bằng chứng thực nghiệm, giữ để tham khảo |

## Bắt đầu nhanh

Cần Git và [uv](https://docs.astral.sh/uv/). Lệnh dưới đây chạy được trên Windows, Linux và macOS:

```bash
git clone https://github.com/Nguyen-Hao-Khang/vla-robot-sorting.git
cd vla-robot-sorting
uv sync
uv run pytest
```

Môi trường mô phỏng RoboTwin chỉ chạy trên Ubuntu 22.04 có GPU NVIDIA. Hướng dẫn đầy đủ: [docs/cai-dat-moi-truong.md](docs/cai-dat-moi-truong.md).

## Tài liệu

| Tài liệu | Nội dung |
|---|---|
| [docs/kien-truc-he-thong.md](docs/kien-truc-he-thong.md) | Kiến trúc hệ thống và luồng dữ liệu |
| [docs/cai-dat-moi-truong.md](docs/cai-dat-moi-truong.md) | Cài đặt môi trường cho từng loại máy |
| [docs/dac-ta-nhiem-vu.md](docs/dac-ta-nhiem-vu.md) | Cảnh, vật, câu lệnh và tiêu chí thành công |
| [docs/du-lieu-va-huan-luyen.md](docs/du-lieu-va-huan-luyen.md) | Sinh dữ liệu, định dạng dataset, huấn luyện |
| [docs/giao-thuc-danh-gia.md](docs/giao-thuc-danh-gia.md) | Chỉ số, số lần thử, cách báo cáo kết quả |
| [docs/adr/](docs/adr/) | Các quyết định kỹ thuật và lý do |
| [docs/project-plan.md](docs/project-plan.md) | Kế hoạch dự án theo giai đoạn |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Quy trình làm việc, quy ước nhánh, commit, code, báo cáo |
| [report/README.md](report/README.md) | Quy tắc viết và biên dịch báo cáo |

## Giấy phép

Mã nguồn phát hành theo [giấy phép MIT](LICENSE).
