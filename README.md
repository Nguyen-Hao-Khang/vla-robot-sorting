# VLA Robot Sorting

Phân loại vật thể bằng cánh tay robot Panthera-HT và một mô hình Vision-Language-Action (VLA) nhỏ. Người vận hành ra lệnh bằng tiếng Việt hoặc tiếng Anh, ví dụ "gắp khối màu đỏ bỏ vào khay bên trái". Mô hình nhận ảnh camera, câu lệnh và trạng thái khớp, rồi xuất chuỗi lệnh khớp để gắp và đặt vật vào đúng khay.

[![CI](https://github.com/Nguyen-Hao-Khang/vla-robot-sorting/actions/workflows/ci.yml/badge.svg)](https://github.com/Nguyen-Hao-Khang/vla-robot-sorting/actions/workflows/ci.yml)

| | |
|---|---|
| Đề tài | Đồ án HK261, Khoa Khoa học và Kỹ thuật Máy tính, Trường Đại học Bách khoa – ĐHQG-HCM |
| Giảng viên hướng dẫn | TS. Lê Trọng Nhân |
| Nhóm | Văn Quốc Huy, Nguyễn Hảo Khang, Phan Gia Phúc |
| Robot | Panthera-HT, 6 khớp, kẹp hai ngón |
| Mô phỏng | RoboTwin (SAPIEN, PhysX), CuRobo v0.7.8 |
| Mô hình | VLA nhỏ tinh chỉnh bằng LeRobot (SmolVLA hoặc Evo-1, [ADR 0005](docs/adr/0005-vla-model-selection.md)); ACT làm mô hình đối chứng |
| Ngôn ngữ | Python 3.10–3.12, quản lý bằng uv |

## Mục tiêu

1. Tinh chỉnh một mô hình VLA nhỏ, suy luận được trên GPU 6 GB, phân loại vật theo màu và hình dạng từ lệnh tiếng Việt và tiếng Anh.
2. Sinh dữ liệu huấn luyện trong mô phỏng: chuyên gia mô phỏng thực hiện thao tác, mô hình ngôn ngữ sinh câu lệnh đa dạng.
3. Chạy mô hình trên Panthera-HT thật qua một lớp an toàn giới hạn vùng làm việc, vận tốc và dừng khẩn cấp.
4. Đánh giá định lượng trong mô phỏng và trên robot thật: tỉ lệ thành công, độ hiểu lệnh, độ tin cậy thao tác.

Đặc tả nhiệm vụ và tiêu chí thành công: [`docs/task-spec.md`](docs/task-spec.md).

## Bắt đầu nhanh

Cần Git và [uv](https://docs.astral.sh/uv/). Các lệnh sau chạy được trên Windows, Linux và macOS:

```sh
git clone https://github.com/Nguyen-Hao-Khang/vla-robot-sorting.git
cd vla-robot-sorting
uv sync
uv run pytest
```

Mô phỏng RoboTwin chỉ chạy trên Ubuntu 22.04 có GPU NVIDIA. Huấn luyện cần GPU từ 16 GB hoặc Kaggle. [`docs/getting-started.md`](docs/getting-started.md) hướng dẫn cài đặt cho từng loại máy.

## Cấu trúc thư mục

```
vla-robot-sorting/
├── src/vla_robot_sorting/   Thư viện dự án: giao diện dữ liệu, mô phỏng, an toàn, đánh giá
├── tools/
│   ├── twin/                Mô phỏng tương tác, chuyên gia sinh demo, ghi dữ liệu,
│   │                        chuyển đổi LeRobot, cầu nối chạy policy
│   └── setup/               Script cài đặt và kiểm tra môi trường
├── configs/                 Cấu hình camera mô phỏng
├── notebooks/               Notebook huấn luyện trên Kaggle
├── tests/                   Unit test, chạy không cần GPU hay mô phỏng
├── report/                  Báo cáo khoa học (LaTeX, tiếng Việt)
├── docs/                    Tài liệu kỹ thuật, quyết định kiến trúc, kế hoạch
├── legacy/sprint1/          Pipeline mô-đun giai đoạn đầu, giữ để tham khảo
└── .github/                 CI và mẫu issue
```

## Tài liệu

| Tài liệu | Nội dung |
|---|---|
| [`docs/getting-started.md`](docs/getting-started.md) | Cài đặt môi trường phát triển, mô phỏng, huấn luyện, báo cáo |
| [`docs/architecture.md`](docs/architecture.md) | Thành phần hệ thống, luồng huấn luyện và suy luận, ràng buộc phần cứng |
| [`docs/task-spec.md`](docs/task-spec.md) | Cảnh, vật, câu lệnh, tiêu chí thành công |
| [`docs/data-and-training.md`](docs/data-and-training.md) | Sinh dữ liệu, định dạng dataset, huấn luyện |
| [`docs/evaluation-protocol.md`](docs/evaluation-protocol.md) | Chỉ số, thiết kế thí nghiệm, thống kê, truy vết kết quả |
| [`docs/adr/`](docs/adr/) | Các quyết định kỹ thuật và lý do |
| [`docs/project-plan.md`](docs/project-plan.md) | Kế hoạch theo giai đoạn, điểm kiểm tra, đối chiếu yêu cầu đề tài |
| [`report/README.md`](report/README.md) | Biên dịch báo cáo, quy tắc trích dẫn, hình và bảng |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | Quy ước nhánh, commit, pull request |

## Kiểm thử

Workflow `CI` của GitHub Actions chạy trên mỗi pull request và mỗi lần push lên `main`:

- **Lint and test:** `ruff check .` và `pytest`. Pull request chỉ merge được khi job này xanh.
- **Build report:** biên dịch `report/` khi thư mục này thay đổi và đính kèm `main.pdf` trong mục Artifacts.

Chạy tại máy:

```sh
uv run ruff check .
uv run pytest
```

Test trong `tests/` không cần GPU, mô phỏng hay mạng. Test của `legacy/sprint1/` không chạy mặc định; cách chạy ghi trong [`legacy/sprint1/README.md`](legacy/sprint1/README.md).

## Trạng thái dự án

| Hạng mục | Trạng thái |
|---|---|
| Mô phỏng Panthera-HT trong RoboTwin | Hoạt động |
| Chuyên gia mô phỏng ghi demo (`tools/twin`) | Hoạt động, cần mở rộng ngẫu nhiên hóa cảnh |
| Chuyển dữ liệu sang định dạng LeRobot | Hoạt động |
| Tinh chỉnh mô hình VLA | Đang chọn mô hình |
| Robot thật | Chờ nhận thiết bị, dự kiến tháng 11/2026 |

Kế hoạch đến 31/12/2026: [`docs/project-plan.md`](docs/project-plan.md).

## Giấy phép

Mã nguồn phát hành theo [giấy phép MIT](LICENSE).
