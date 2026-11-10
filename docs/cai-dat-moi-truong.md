# Cài đặt môi trường

## 1. Máy nào làm được việc gì

| Việc | Windows | Ubuntu 22.04 + GPU NVIDIA | Kaggle / Colab |
|---|---|---|---|
| Viết code, chạy test, xử lý dữ liệu, viết tài liệu | Có | Có | Không cần |
| Chạy mô phỏng RoboTwin, sinh demo, đánh giá policy trong mô phỏng | Không (WSL không hỗ trợ) | Có | Không |
| Tinh chỉnh SmolVLA hoặc Evo-1 | Có, nếu GPU ≥ 16 GB | Có, nếu GPU ≥ 16 GB | Có |
| Biên dịch báo cáo LaTeX | Có (TeX Live hoặc MiKTeX) | Có | Không cần |

## 2. Môi trường phát triển (mọi máy)

Cài [Git](https://git-scm.com/) và [uv](https://docs.astral.sh/uv/). uv tự tải Python phù hợp (3.10 đến 3.12).

```bash
git clone https://github.com/Nguyen-Hao-Khang/vla-robot-sorting.git
cd vla-robot-sorting
uv sync
uv run ruff check .
uv run pytest
```

Kết quả mong đợi:

- Ruff báo `All checks passed!`.
- Khoảng 590 test chạy qua.
- Khoảng 20 test được bỏ qua: các test này cần dữ liệu camera thu từ mô phỏng (biến môi trường `S1I_NATIVE_DIR`) hoặc mã nguồn RoboTwin.

Trên Windows, `tools/setup/bootstrap_windows.ps1` chạy toàn bộ các bước trên. `tools/setup/check_windows_env.ps1` in phiên bản Git, Python, uv và GPU để kiểm tra nhanh.

Không đặt thư mục repo trong OneDrive, Dropbox hoặc Google Drive. Các dịch vụ này có thể đồng bộ dở dang thư mục `.git` và làm hỏng repo.

## 3. Môi trường mô phỏng (Ubuntu 22.04)

### 3.1 Yêu cầu

- Ubuntu 22.04 cài trực tiếp trên máy (không dùng máy ảo hay WSL).
- GPU NVIDIA hỗ trợ Vulkan. Dự án đã kiểm thử với driver 595 trên GTX 1660 Ti.
- Khoảng 40 GB ổ trống cho RoboTwin, mô hình 3D và dữ liệu.

### 3.2 Phiên bản cố định

| Thành phần | Nguồn | Phiên bản |
|---|---|---|
| Panthera-HT RoboTwin | `github.com/HighTorque-Robotics/Panthera-HT_RoboTwin` | commit `2aee31ad75cb7b0cd779540441c6025853a5b60d` |
| CuRobo | `github.com/NVlabs/curobo` | `v0.7.8` (commit `d64c4b00`) |
| SAPIEN | PyPI | `3.0.0b1` |
| PyTorch | PyPI | bản dựng CUDA 12.8 |

Danh sách đầy đủ nằm trong `legacy/sprint1/configs/s1.1/repos.lock.yaml`.

### 3.3 Các bước

1. Cài driver và công cụ hệ thống:

   ```bash
   bash tools/setup/bootstrap_ubuntu_host.sh
   ```

   Script cài `build-essential`, `vulkan-tools`, `ffmpeg` và driver NVIDIA do Ubuntu đề xuất, rồi ghi log vào `~/damh-host-bootstrap/`. Khởi động lại máy sau khi cài driver.

2. Cài Miniforge, tạo môi trường conda `RoboTwin`, clone Panthera-HT RoboTwin đúng commit ở trên. Làm theo hướng dẫn cài đặt trong README của repo đó, gồm cả bước tải mô hình 3D.
3. Cài CuRobo `v0.7.8` vào môi trường `RoboTwin` theo hướng dẫn của NVIDIA.
4. Clone repo dự án vào ổ Linux, ví dụ `~/projects/vla-robot-sorting`. Không đặt repo trên phân vùng NTFS.
5. Chạy thử mô phỏng:

   ```bash
   conda activate RoboTwin
   cd ~/projects/vla-robot-sorting
   python tools/twin/twin_console.py \
       --robotwin-root ~/worktrees/Panthera-HT_RoboTwin/RoboTwin \
       --level 1
   ```

   Cửa sổ SAPIEN mở ra với cánh tay Panthera-HT và một khối trên bàn. Log phiên chạy ghi vào `~/twin_logs/`, đổi bằng `--log-dir`.

### 3.4 Khóa API mô hình ngôn ngữ

Một số công cụ gọi Gemini, ví dụ để sinh câu lệnh hoặc lập kế hoạch trong mô phỏng tương tác. Khóa API chỉ đọc từ biến môi trường `GEMINI_API_KEY`.

```bash
mkdir -p ~/.config/damh
printf 'GEMINI_API_KEY=<khóa của bạn>\n' > ~/.config/damh/gemini.env
chmod 600 ~/.config/damh/gemini.env
set -a; . ~/.config/damh/gemini.env; set +a
```

Không ghi khóa vào code, notebook, issue hay tin nhắn.

## 4. Môi trường huấn luyện

### 4.1 Máy có GPU NVIDIA (Windows hoặc Linux)

```bash
uv venv .venv-train --python 3.10
uv pip install --python .venv-train "lerobot[smolvla]"
```

GPU thế hệ RTX 50 cần PyTorch dựng cho CUDA 12.8 trở lên. Kiểm tra:

```bash
.venv-train/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Trên Windows, đường dẫn là `.venv-train\Scripts\python`.

### 4.2 Kaggle

1. Tạo tài khoản Kaggle và xác minh số điện thoại để dùng GPU (30 giờ mỗi tuần).
2. Tải `notebooks/kaggle_train_panthera_twin.ipynb` lên Kaggle, gắn dataset, chọn GPU T4 x2.
3. Lưu checkpoint vào Kaggle Output hoặc Hugging Face Hub. Không commit checkpoint vào repo.

## 5. Báo cáo LaTeX

Cần TeX Live (Linux, Windows) hoặc MiKTeX (Windows) có gói `vntex`.

```bash
cd report
latexmk -pdf main.tex
```

Quy tắc viết báo cáo: [report/README.md](../report/README.md).
