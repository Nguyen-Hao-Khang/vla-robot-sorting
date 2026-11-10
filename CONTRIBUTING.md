# Hướng dẫn đóng góp

## Quy trình

1. Mỗi đầu việc trong [kế hoạch](docs/project-plan.md) có một issue. Gán issue cho mình trước khi làm.
2. Tạo nhánh từ `main`: `git switch -c <ten>/<viec>`, ví dụ `khang/scene-randomizer`, `phuc/chapter-2`.
3. Mở pull request vào `main`, ghi `Closes #<số issue>`. Tự kiểm tra thay đổi của mình. Khi CI xanh, tự bấm **Squash and merge**, không cần chờ người khác duyệt.

Không push thẳng lên `main`.

## Commit và tiêu đề pull request

Viết bằng tiếng Anh theo mẫu [Conventional Commits](https://www.conventionalcommits.org/): `<type>(<scope>): <summary>`.

- `type`: `feat`, `fix`, `docs`, `test`, `refactor`, `exp` (thí nghiệm, kết quả), `chore` (công cụ, CI).
- `scope` (không bắt buộc): `sim`, `data`, `train`, `eval`, `robot`, `safety`, `report`.
- Ví dụ: `feat(data): add randomized scene generator`. Tránh `update`, `fix bug`, `WIP`.

Tài liệu, báo cáo, mô tả issue và pull request viết tiếng Việt. Code, comment, commit và tên nhánh viết tiếng Anh.

## Trước khi push

```bash
uv run ruff check .
uv run pytest
```

- Thay đổi hành vi thì thêm test. Test không được cần GPU, mô phỏng hay mạng.
- Không nới ngưỡng an toàn để một lần chạy qua được.
- Không ghi đường dẫn tuyệt đối của máy cá nhân trong code.

## Không đưa vào repo

- API key, token, file `.env`.
- Dataset, checkpoint, video, log huấn luyện: lưu trên Hugging Face Hub hoặc Kaggle, dán liên kết vào pull request.
- PDF báo cáo và các file build khác.

## Kết quả thực nghiệm và báo cáo

- Mỗi số liệu trong báo cáo phải truy được về commit, file cấu hình, seed và log gốc ([giao thức đánh giá](docs/giao-thuc-danh-gia.md)).
- Quy tắc trích dẫn, hình, bảng và phân công chương: [report/README.md](report/README.md).
