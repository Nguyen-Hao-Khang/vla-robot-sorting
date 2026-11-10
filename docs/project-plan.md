# Kế hoạch dự án

Thời gian: 10/11/2026 – 31/12/2026. Bảng kế hoạch đầy đủ (có cột người thực hiện, người review, trạng thái) được quản lý trên Google Sheets; file này là bản tham chiếu trong repo. Mỗi đầu việc ứng với một issue trên GitHub.

## 1. Phạm vi

| Mức | Nội dung |
|---|---|
| Bắt buộc | Mô hình VLA nhỏ đã tinh chỉnh, phân loại vật theo màu và hình dạng từ lệnh tiếng Việt và tiếng Anh, chạy trong mô phỏng RoboTwin và trên Panthera-HT thật qua lớp an toàn. Đánh giá định lượng ≥ 30 lượt mỗi điều kiện. Báo cáo 6 chương. |
| Mở rộng (nếu còn thời gian) | Phân loại theo nhóm ngữ nghĩa trong mô phỏng. |
| Ngoài phạm vi | Kết hợp mô hình thị giác–ngôn ngữ lập kế hoạch với VLA; vật biến dạng, vật xếp chồng; lệnh nhiều bước. |

## 2. Giai đoạn

| Giai đoạn | Thời gian | Kết quả cuối giai đoạn |
|---|---|---|
| 1. Dữ liệu mô phỏng và chọn mô hình | 10/11 – 23/11 | Có bộ sinh cảnh ngẫu nhiên, bộ sinh câu lệnh song ngữ, script đánh giá tự động; chọn xong mô hình VLA nền; nộp bản nháp Chương 1 và Chương 2. |
| 2. Huấn luyện và đánh giá trong mô phỏng | 24/11 – 07/12 | Có dataset mô phỏng 500–1.000 demo, mô hình chính và mô hình đối chứng đã huấn luyện, kết quả đánh giá trong mô phỏng. |
| 3. Tích hợp robot thật | 24/11 – 07/12 | Robot thật điều khiển được từ máy tính qua lớp an toàn, có camera đã hiệu chuẩn và 30–50 demo thật. |
| 4. Thực nghiệm trên robot thật | 08/12 – 20/12 | Có kết quả định lượng trên robot thật, phân tích lỗi và video demo. Đóng băng kết quả ngày 20/12. |
| 5. Hoàn thiện báo cáo | 14/12 – 31/12 | Báo cáo đủ 6 chương đã review chéo, slide, nộp ngày 31/12. |

Giai đoạn 2 và 3 chạy song song: giai đoạn 2 dùng máy huấn luyện và máy mô phỏng, giai đoạn 3 dùng robot thật.

## 3. Điểm kiểm tra

Tại mỗi điểm, nhóm kiểm tra điều kiện. Nếu chưa đạt, áp dụng phương án dự phòng thay vì lùi hạn.

| Ngày | Điểm kiểm tra | Điều kiện đạt | Nếu chưa đạt |
|---|---|---|---|
| 16/11 | Gửi GVHD Chương 1 và Chương 2 | Bản nháp biên dịch được, đã review chéo | Gửi bản hiện có, ghi rõ phần còn thiếu |
| 23/11 | Chốt mô hình và bộ sinh dữ liệu | ADR 0005 được chấp nhận; sinh được 50 demo có câu lệnh song ngữ | Dùng SmolVLA làm mặc định và tiếp tục so sánh song song |
| 30/11 | Robot thật hoạt động | Robot điều khiển được qua lớp an toàn; camera đã lắp | Báo GVHD; giai đoạn 4 giảm còn hai chế độ (chỉ mô phỏng, kết hợp) với số lượt tối thiểu |
| 07/12 | Có kết quả trong mô phỏng | Bảng kết quả thí nghiệm 2.4 và 2.5 | Bỏ đầu việc mở rộng 4.5; dồn GPU cho mô hình chính |
| 20/12 | Đóng băng kết quả | Mọi bảng và hình sinh được từ log | Không chạy thí nghiệm mới sau ngày này; ghi phần thiếu vào mục Hạn chế |
| 31/12 | Nộp báo cáo | Release v1.0 trên GitHub | — |

## 4. Đầu việc

### Giai đoạn 1. Dữ liệu mô phỏng và chọn mô hình (10/11 – 23/11)

| STT | Hạng mục | Công việc | Kết quả bàn giao | Tiêu chí hoàn thành | Phụ thuộc | Bắt đầu | Hạn |
|---|---|---|---|---|---|---|---|
| 1.1 | Quản lý | Thiết lập làm việc nhóm | Issue trên GitHub | Mỗi đầu việc giai đoạn 1 có issue và người thực hiện | — | 10/11 | 12/11 |
| 1.2 | Dữ liệu | Chốt đặc tả nhiệm vụ | docs/dac-ta-nhiem-vu.md | Cả nhóm duyệt qua pull request | — | 10/11 | 13/11 |
| 1.3 | Mô phỏng | Ngẫu nhiên hóa cảnh khi ghi demo (Cần máy Ubuntu có GPU) | tools/twin/twin_record.py, test | Sinh 50 demo với seed cố định, tái tạo được; test chạy qua | 1.2 | 12/11 | 19/11 |
| 1.4 | Dữ liệu | Bộ sinh câu lệnh song ngữ (Làm được trên Windows hoặc đám mây) | Script trong src/ hoặc tools/, file câu lệnh | ≥ 30 câu mỗi cặp cho mỗi ngôn ngữ; test chạy qua | 1.2 | 12/11 | 19/11 |
| 1.5 | Dữ liệu | Tập câu lệnh kiểm tra do người viết | File câu lệnh kiểm tra | 100–200 câu, tách riêng khỏi dữ liệu huấn luyện | 1.2 | 13/11 | 23/11 |
| 1.6 | Mô hình | Môi trường huấn luyện | Hướng dẫn cập nhật trong docs/cai-dat-moi-truong.md | Mỗi mô hình chạy được 100 bước huấn luyện | — | 10/11 | 16/11 |
| 1.7 | Mô hình | So sánh và chọn mô hình VLA nền | ADR 0005 có bảng số đo và quyết định | ADR 0005 chuyển sang trạng thái Chấp nhận | 1.3, 1.6, 1.8 | 17/11 | 23/11 |
| 1.8 | Đánh giá | Script đánh giá tự động trong mô phỏng | Script đánh giá, test | Chạy được với policy và chuyên gia mô phỏng; test chạy qua | 1.2 | 13/11 | 21/11 |
| 1.9 | Báo cáo | Chương 1: Giới thiệu | report/chapters/01-introduction.tex | Bản nháp biên dịch được, có trích dẫn | — | 12/11 | 15/11 |
| 1.10 | Báo cáo | Chương 2: Cơ sở lý thuyết và công trình liên quan | report/chapters/02-background.tex, references.bib | Bản nháp biên dịch được; mọi trích dẫn có DOI hoặc arXiv | — | 12/11 | 15/11 |
| 1.11 | Báo cáo | Review chéo Chương 1, 2 và gửi GVHD (GVHD yêu cầu nộp cuối tuần 15–16/11) | Bản PDF gửi GVHD | Pull request được duyệt và merge; PDF đã gửi | 1.9, 1.10 | 15/11 | 16/11 |

### Giai đoạn 2. Huấn luyện và đánh giá trong mô phỏng (24/11 – 07/12)

| STT | Hạng mục | Công việc | Kết quả bàn giao | Tiêu chí hoàn thành | Phụ thuộc | Bắt đầu | Hạn |
|---|---|---|---|---|---|---|---|
| 2.1 | Dữ liệu | Dataset mô phỏng phiên bản 1 | Dataset panthera_twin_v1 trên Hugging Face Hub | 500–1.000 lượt thành công; thống kê dataset trong README của dataset | 1.3, 1.4 | 24/11 | 28/11 |
| 2.2 | Mô hình | Tinh chỉnh mô hình chính | Checkpoint, file cấu hình, log huấn luyện | Checkpoint chạy được qua policy_bridge trong mô phỏng | 1.7, 2.1 | 28/11 | 03/12 |
| 2.3 | Mô hình | Huấn luyện mô hình đối chứng ACT | Checkpoint, file cấu hình, log huấn luyện | Checkpoint chạy được trong mô phỏng | 2.1 | 28/11 | 03/12 |
| 2.4 | Đánh giá | Thí nghiệm trong mô phỏng | Log đánh giá, bảng kết quả sinh bằng script | ≥ 30 lượt mỗi điều kiện, có khoảng tin cậy | 1.5, 1.8, 2.2 | 03/12 | 07/12 |
| 2.5 | Đánh giá | So sánh với mô hình đối chứng | Log đánh giá, bảng so sánh | ≥ 30 lượt mỗi mô hình | 1.8, 2.2, 2.3 | 03/12 | 07/12 |

### Giai đoạn 3. Tích hợp robot thật (24/11 – 07/12)

| STT | Hạng mục | Công việc | Kết quả bàn giao | Tiêu chí hoàn thành | Phụ thuộc | Bắt đầu | Hạn |
|---|---|---|---|---|---|---|---|
| 3.1 | Robot | Nhận robot và kết nối SDK (Phụ thuộc ngày giao robot) | Script kết nối, ghi chú trong docs/ | Đọc và điều khiển được từng khớp ở tốc độ thấp | — | 24/11 | 28/11 |
| 3.2 | An toàn | Lớp an toàn cho robot thật (Phần cấu hình và test làm trước khi có robot) | Cấu hình và test trong src/vla_robot_sorting/safety/ | Test chặn lệnh ngoài vùng chạy qua; thử trên robot thật ở tốc độ thấp | 3.1 | 24/11 | 01/12 |
| 3.3 | Robot | Camera và hiệu chuẩn | File hiệu chuẩn, configs/twin_cameras.json cập nhật | Ghi được sai số hiệu chuẩn | 3.1 | 28/11 | 03/12 |
| 3.4 | Dữ liệu | Thu demo thật | Dataset panthera_real_v1 trên Hugging Face Hub | 30–50 lượt thành công | 3.1, 3.3 | 01/12 | 07/12 |
| 3.5 | Robot | Chạy policy trên robot thật | Script điều khiển, video chạy thử | Robot thực hiện trọn một lượt gắp–đặt do policy điều khiển | 2.2, 3.2 | 03/12 | 07/12 |
| 3.6 | Báo cáo | Chương 3: Phần cứng và hệ thu nhận | report/chapters/03-hardware.tex | Bản nháp được review chéo | 3.2, 3.3 | 01/12 | 10/12 |

### Giai đoạn 4. Thực nghiệm trên robot thật (08/12 – 20/12)

| STT | Hạng mục | Công việc | Kết quả bàn giao | Tiêu chí hoàn thành | Phụ thuộc | Bắt đầu | Hạn |
|---|---|---|---|---|---|---|---|
| 4.1 | Mô hình | Huấn luyện với dữ liệu thật | Hai checkpoint, file cấu hình, log | Cả hai checkpoint chạy được qua lớp an toàn | 2.1, 3.4 | 08/12 | 11/12 |
| 4.2 | Đánh giá | Thí nghiệm trên robot thật (Giảm số chế độ nếu robot về muộn) | Phiếu đánh giá, video, bảng kết quả | ≥ 30 lượt mỗi chế độ, có khoảng tin cậy | 3.5, 4.1 | 11/12 | 18/12 |
| 4.3 | Đánh giá | Phân tích lỗi và sinh hình, bảng | Script và hình, bảng trong report/figures | Mọi số liệu trong báo cáo truy được về log | 2.4, 2.5, 4.2 | 15/12 | 20/12 |
| 4.4 | Robot | Video demo | Video ≤ 3 phút | Nhóm duyệt | 4.2 | 15/12 | 20/12 |
| 4.5 | Mô hình | Mở rộng: phân loại theo nhóm ngữ nghĩa (Chỉ làm khi giai đoạn 2 xong đúng hạn) | Dataset, checkpoint, bảng kết quả | ≥ 30 lượt trong mô phỏng | 2.4 | 08/12 | 20/12 |
| 4.6 | Báo cáo | Chương 4: Phương pháp | report/chapters/04-method.tex | Bản nháp được review chéo | 2.2, 3.5 | 01/12 | 14/12 |

### Giai đoạn 5. Hoàn thiện báo cáo (14/12 – 31/12)

| STT | Hạng mục | Công việc | Kết quả bàn giao | Tiêu chí hoàn thành | Phụ thuộc | Bắt đầu | Hạn |
|---|---|---|---|---|---|---|---|
| 5.1 | Báo cáo | Chương 5: Thực nghiệm và đánh giá | report/chapters/05-experiments.tex | Bản nháp được review chéo | 4.3 | 16/12 | 24/12 |
| 5.2 | Báo cáo | Chương 6 và Tóm tắt | 06-conclusion.tex, 00-abstract.tex | Bản nháp được review chéo | 5.1 | 21/12 | 26/12 |
| 5.3 | Báo cáo | Review chéo toàn văn | Danh sách sửa đổi trong pull request | Không còn góp ý mở | 5.2 | 26/12 | 29/12 |
| 5.4 | Báo cáo | Slide bảo vệ | File slide | Nhóm tập trình bày một lần | 5.1 | 26/12 | 30/12 |
| 5.5 | Quản lý | Nộp báo cáo và phát hành | Release v1.0 | Đã nộp đúng hạn | 5.3, 5.4 | 31/12 | 31/12 |

## 5. Đối chiếu yêu cầu đề tài

| Yêu cầu | Nội dung | Đầu việc | Phần báo cáo |
|---|---|---|---|
| 1 | Tổng quan VLA, mô hình nền tảng, cơ chế ánh xạ lệnh thành hành động | 1.10 | Chương 2 |
| 2 | Phần cứng, giao thức điều khiển, hệ thu ảnh | 3.1, 3.2, 3.3, 3.6 | Chương 3 |
| 3 | Triển khai và tinh chỉnh VLA cho nhận diện, định vị, gắp–đặt theo lệnh | 1.3–1.7, 2.1–2.3, 4.1 | Chương 4 |
| 4 | Ứng dụng phân loại hoàn chỉnh tích hợp VLA với cánh tay robot | 3.5, 4.2, 4.4 | Chương 4, 5; video demo |
| 5 | Đánh giá nhận diện, hiểu ngôn ngữ, độ tin cậy thao tác | 1.8, 2.4, 2.5, 4.2, 4.3 | Chương 5 |

## 6. Rủi ro

| Rủi ro | Ảnh hưởng | Cách xử lý |
|---|---|---|
| Robot về sau 30/11 | Ít thời gian thu demo và thí nghiệm thật | Phần lớp an toàn và cầu nối được viết và kiểm thử trong mô phỏng trước; giảm số chế độ thí nghiệm thật |
| Hết quota GPU miễn phí | Chậm huấn luyện | Ưu tiên RTX 5090; chi tối đa 500.000 đ thuê GPU đám mây |
| Mô hình hiểu kém lệnh tiếng Việt | Tỉ lệ thành công thấp với lệnh tiếng Việt | Báo cáo trung thực kết quả; tăng tỉ lệ câu lệnh tiếng Việt trong dữ liệu huấn luyện |
| Máy mô phỏng chỉ dùng được 2–3 buổi mỗi tuần | Chậm sinh dữ liệu và đánh giá | Chạy sinh dữ liệu và đánh giá không giao diện, theo lô qua đêm |
