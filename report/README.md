# Báo cáo

Báo cáo khoa học của đề tài, viết bằng LaTeX, tiếng Việt.

## Biên dịch

```bash
cd report
latexmk -pdf main.tex
```

Cần TeX Live hoặc MiKTeX có gói `vntex`. CI biên dịch báo cáo ở mỗi pull request có thay đổi trong `report/` và đính kèm `main.pdf` trong mục Artifacts của lần chạy. Không commit file PDF.

## Cấu trúc

| File | Nội dung |
|---|---|
| `main.tex` | File gốc: thông tin môn học, thứ tự chương |
| `setup/` | Gói, định dạng, bảng thuật ngữ, trang bìa |
| `chapters/01-introduction.tex` | Chương 1. Giới thiệu |
| `chapters/02-background.tex` | Chương 2. Cơ sở lý thuyết và công trình liên quan |
| `chapters/03-hardware.tex` | Chương 3. Phần cứng và hệ thu nhận |
| `chapters/04-method.tex` | Chương 4. Phương pháp |
| `chapters/05-experiments.tex` | Chương 5. Thực nghiệm và kết quả |
| `chapters/06-conclusion.tex` | Chương 6. Thảo luận và kết luận |
| `references.bib` | Tài liệu tham khảo |
| `figures/` | Hình ảnh |

## Phân công

Mỗi chương có một người viết và một người review chéo. Phân công cụ thể ghi trong kế hoạch dự án (`docs/project-plan.md` và bảng kế hoạch). Người review kiểm tra:

- mỗi khẳng định có trích dẫn hoặc số liệu đi kèm;
- mỗi số liệu khớp với log gốc;
- thuật ngữ thống nhất với bảng thuật ngữ trong `setup/glossaries.tex`.

## Trích dẫn

1. Mỗi mục trong `references.bib` có DOI hoặc mã arXiv. Tài liệu không có cả hai (sách, tài liệu kỹ thuật của hãng) ghi URL và ngày truy cập trong trường `note`.
2. Người thêm trích dẫn phải đã đọc bản gốc, không trích lại qua bài khác hay qua bản tóm tắt.
3. Khóa trích dẫn theo mẫu `<họ tác giả đầu><năm><từ đầu tiêu đề>`, ví dụ `wen2025tinyvla`.
4. Không trích dẫn nguồn không kiểm chứng được (blog không rõ tác giả, nội dung do AI sinh).

## Hình và bảng

- Mọi biểu đồ và bảng số liệu được sinh bằng script trong repo từ log đánh giá. Ghi tên script trong comment ngay trên lệnh `\includegraphics` hoặc `\input`.
- Sơ đồ vẽ tay lưu kèm file nguồn (`.drawio`, `.svg`) trong `figures/`.
- Ảnh chụp màn hình và ảnh robot ghi rõ nguồn trong chú thích.
- Hình dùng định dạng vector (PDF, SVG) khi có thể. Ảnh raster tối thiểu 300 dpi ở kích thước in.

## Văn phong

- Viết ngôi thứ nhất số nhiều ("chúng tôi") hoặc câu không chủ ngữ, thống nhất trong toàn bài.
- Thuật ngữ tiếng Anh chưa có từ tương đương phổ biến thì giữ nguyên, in nghiêng ở lần đầu và đưa vào bảng thuật ngữ.
- Mỗi đoạn có một ý chính. Không dùng từ cảm thán hay quảng cáo.
