# Đóng góp

Tài liệu này quy định cách nhóm đặt tên nhánh, viết commit, viết comment và mở pull request cho dự án.

## 1. Ngôn ngữ

| Thành phần | Ngôn ngữ |
|---|---|
| Tài liệu (`README`, `docs/`), báo cáo (`report/`), nội dung issue và pull request | Tiếng Việt |
| Tên biến, hàm, lớp, comment, docstring, log, commit, tên nhánh, tiêu đề issue và pull request | Tiếng Anh |

## 2. Nhánh

`main` luôn qua được lint và test. Mỗi việc làm trên một nhánh ngắn hạn và vào `main` qua pull request. Không push thẳng lên `main`.

Tên nhánh: `<người-phụ-trách>/<việc>`

| Phần | Quy tắc |
|---|---|
| `người-phụ-trách` | Tên định danh của thành viên chịu trách nhiệm: `khangnh`, `huyvq`, `phucpg` |
| `việc` | 2 đến 5 từ tiếng Anh viết thường, nối bằng dấu gạch ngang, chỉ ký tự ASCII |

Ví dụ: `khangnh/scene-randomizer`, `huyvq/smolvla-benchmark`, `phucpg/chapter-2-background`.

## 3. Commit

Commit theo [Conventional Commits](https://www.conventionalcommits.org/), viết bằng tiếng Anh:

```
<type>(<scope>): <summary>

<body: what and why, wrapped at 72 columns>

Refs: #<issue>
```

- `summary`: thể mệnh lệnh, chữ thường, không dấu chấm cuối, tối đa 50 ký tự.
- `body`: tùy chọn, giải thích lý do thay đổi thay vì kể lại diff.
- Mỗi commit chứa một thay đổi logic. Lint và test phải qua sau mỗi commit.

| `type` | Dùng khi |
|---|---|
| `feat` | Thêm tính năng |
| `fix` | Sửa lỗi |
| `test` | Thêm hoặc sửa test |
| `docs` | Tài liệu hoặc nội dung báo cáo |
| `exp` | Cấu hình thí nghiệm, kết quả đánh giá, bảng số liệu |
| `build` | Thư viện phụ thuộc, CI, công cụ |
| `refactor` | Đổi cấu trúc mã, không đổi hành vi |

Scope thường dùng: `sim`, `data`, `train`, `eval`, `robot`, `safety`, `report`.

Ví dụ:

```
feat(data): randomize object layout per episode
fix(sim): add table objects to curobo world
exp(eval): add smolvla simulation results
docs(report): draft related work on small vlas
```

## 4. Mã nguồn

- Định dạng và lint bằng `ruff` (độ dài dòng 100).
- Comment và docstring viết tiếng Anh, ngắn, nêu ý định hoặc ràng buộc thay vì lặp lại mã.
- Thay đổi hành vi thì thêm test. Test không được cần GPU, mô phỏng hay mạng; phần cần phần cứng đánh dấu `@pytest.mark.simulator` hoặc `@pytest.mark.network`.
- Không nới ngưỡng an toàn để một lần chạy qua được.
- Không ghi đường dẫn tuyệt đối của máy cá nhân. Dùng tham số dòng lệnh hoặc biến môi trường.
- Việc dự định làm: `TODO(<scope>): <action>`.

## 5. Pull request

Trước khi mở pull request, chạy:

```sh
uv run ruff check .
uv run pytest
```

Quy trình:

1. Cập nhật `main` và tạo nhánh mới từ đó.
2. Commit theo mục 3, push nhánh lên GitHub.
3. Mở pull request vào `main`. Mô tả nêu việc đã làm, cách kiểm tra, ghi `Closes #<số issue>`.
4. Tự kiểm tra thay đổi của mình. Khi CI xanh, tự bấm **Squash and merge**. Không cần chờ người khác duyệt.
5. GitHub tự xóa nhánh sau khi merge.

## 6. Dữ liệu và kết quả

Không commit:

- API key, token, file `.env`;
- dataset, checkpoint, video, log huấn luyện: lưu trên Hugging Face Hub hoặc Kaggle, dán liên kết vào pull request;
- PDF báo cáo và các file build khác.

Mỗi số liệu đưa vào báo cáo phải truy được về commit, file cấu hình, seed và log gốc theo [`docs/evaluation-protocol.md`](docs/evaluation-protocol.md). Quy tắc trích dẫn, hình và bảng: [`report/README.md`](report/README.md).
