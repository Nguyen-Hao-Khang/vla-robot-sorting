# Hướng dẫn đóng góp

Tài liệu này quy định cách cả nhóm làm việc trên repo. Mọi thành viên áp dụng cùng một quy trình để lịch sử commit đọc được và mỗi kết quả trong báo cáo truy được về mã nguồn.

## 1. Ngôn ngữ

| Thành phần | Ngôn ngữ |
|---|---|
| Tài liệu (`README`, `docs/`, `CONTRIBUTING`), báo cáo (`report/`), mô tả issue và pull request | Tiếng Việt |
| Tên biến, hàm, lớp, comment trong code, log, commit message, tên nhánh, tiêu đề issue và pull request | Tiếng Anh |

## 2. Quy trình một đầu việc

1. **Nhận việc.** Mỗi đầu việc trong [kế hoạch](docs/project-plan.md) ứng với một issue trên GitHub. Bạn gán issue cho mình trước khi bắt đầu.
2. **Tạo nhánh** từ `main` mới nhất:

   ```bash
   git switch main
   git pull
   git switch -c <ten>/<viec>
   ```

3. **Commit** từng bước nhỏ. Mỗi commit phải qua được lint và test.
4. **Mở pull request** vào `main`. Ghi `Closes #<số issue>` trong phần mô tả.
5. **Review.** Một thành viên khác review và chấp thuận. Tác giả không tự merge pull request của mình.
6. **Merge** bằng **Squash and merge**. GitHub tự xóa nhánh sau khi merge.

Không ai push thẳng lên `main`.

## 3. Đặt tên nhánh

Mẫu: `<ten>/<viec>`

- `<ten>` là tên gọi của người thực hiện, viết thường, không dấu: `khang`, `huy`, `phuc`.
- `<viec>` là mô tả ngắn bằng tiếng Anh, viết thường, nối bằng gạch ngang.

Ví dụ: `khang/scene-randomizer`, `huy/smolvla-benchmark`, `phuc/chapter-2-related-work`.

## 4. Commit message

Repo dùng chuẩn [Conventional Commits](https://www.conventionalcommits.org/).

```
<type>(<scope>): <summary>

<body: what changed and why>

Refs #<issue>
```

| `type` | Dùng khi |
|---|---|
| `feat` | Thêm chức năng |
| `fix` | Sửa lỗi |
| `docs` | Tài liệu hoặc nội dung báo cáo |
| `test` | Chỉ thêm hoặc sửa test |
| `refactor` | Đổi cấu trúc code, không đổi hành vi |
| `exp` | Cấu hình thí nghiệm, kết quả đánh giá, bảng số liệu |
| `chore` | Công cụ, thư viện phụ thuộc, cấu hình repo, CI |

| `scope` | Phạm vi |
|---|---|
| `sim` | Môi trường mô phỏng RoboTwin |
| `data` | Sinh và xử lý dữ liệu |
| `train` | Huấn luyện mô hình |
| `eval` | Đánh giá |
| `robot` | Robot thật, SDK, camera |
| `safety` | Lớp an toàn |
| `report` | Báo cáo LaTeX |
| `docs` | Tài liệu trong `docs/` |
| `ci` | GitHub Actions |

Quy tắc:

- `summary` viết tiếng Anh, thể mệnh lệnh, chữ thường, không quá 72 ký tự, không có dấu chấm cuối.
  - Đúng: `feat(data): add randomized scene generator`
  - Sai: `update`, `fix bug`, `WIP`, `Khang/repo conventions (#2)`
- `body` giải thích lý do thay đổi. Phần diff đã cho thấy nội dung thay đổi.
- Tiêu đề pull request theo đúng mẫu trên. Khi squash merge, tiêu đề này trở thành commit trên `main`.

## 5. Quy tắc code

- Python 3.10 đến 3.12.
- Định dạng và lint bằng `ruff` (độ dài dòng 100).
- Chạy hai lệnh sau trước khi push:

  ```bash
  uv run ruff check .
  uv run pytest
  ```

- **Test:** mỗi thay đổi hành vi đi kèm test. Test không được cần GPU, mô phỏng hay mạng. Phần cần phần cứng đánh dấu `@pytest.mark.simulator` hoặc `@pytest.mark.network`.
- **Thư viện ngoài:** RoboTwin, CuRobo, LeRobot và API mô hình ngôn ngữ phải nằm sau các giao diện trong `src/vla_robot_sorting/contracts`.
- **An toàn:** không nới ngưỡng an toàn để một lần chạy qua được.
- **Dữ liệu mô phỏng:** thông tin lấy trực tiếp từ simulator (vị trí, danh tính vật) được gắn nhãn là ground truth và không được trình bày như kết quả tri giác.
- **Đường dẫn:** không ghi đường dẫn tuyệt đối của máy cá nhân trong code. Dùng tham số dòng lệnh hoặc biến môi trường.

## 6. Những gì không đưa vào repo

| Loại | Lưu ở đâu |
|---|---|
| API key, token, file `.env` | Biến môi trường trên máy của từng người |
| Dataset, checkpoint, video, log huấn luyện | Hugging Face Hub hoặc Kaggle Datasets; dán liên kết vào pull request |
| File sinh ra khi build (PDF báo cáo, `__pycache__`, output notebook) | Không lưu |
| Ghi chú cá nhân, nội dung trao đổi, bản nháp không thuộc code hay báo cáo | Không lưu |

GitHub bật chặn push khi phát hiện khóa bí mật. Nếu lỡ commit một khóa, báo ngay cho cả nhóm và thu hồi khóa đó.

## 7. Kết quả thực nghiệm

Một số liệu chỉ được đưa vào báo cáo khi truy được về:

1. commit sinh ra kết quả;
2. file cấu hình và seed;
3. log hoặc file đánh giá gốc (lưu ngoài Git, có liên kết trong pull request).

Cách đo và cách báo cáo số liệu theo [docs/giao-thuc-danh-gia.md](docs/giao-thuc-danh-gia.md).

## 8. Báo cáo

- Báo cáo nằm trong `report/`, viết bằng LaTeX, tiếng Việt.
- Mỗi chương có một người chịu trách nhiệm viết và một người review chéo. Phân công ghi trong kế hoạch dự án.
- Quy tắc trích dẫn, hình, bảng và cách biên dịch: xem [report/README.md](report/README.md).
- CI biên dịch báo cáo ở mỗi pull request có thay đổi trong `report/` và đính kèm file PDF để review.
