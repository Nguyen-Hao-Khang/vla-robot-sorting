#!/usr/bin/env bash
set -euo pipefail

if [[ "${EUID}" -eq 0 ]]; then
  echo "Run as your normal user; the script uses sudo where required."
  exit 2
fi

. /etc/os-release
if [[ "${ID}" != "ubuntu" || "${VERSION_ID}" != "22.04" ]]; then
  echo "Expected native Ubuntu 22.04; found: ${PRETTY_NAME:-unknown}"
  exit 3
fi

LOG_DIR="$HOME/damh-host-bootstrap"
mkdir -p "$LOG_DIR"
exec > >(tee "$LOG_DIR/bootstrap.log") 2>&1

echo "=== Host bootstrap started: $(date --iso-8601=seconds) ==="
uname -a
lsblk -f
df -h


echo "=== APT update / upgrade ==="
sudo apt update
sudo DEBIAN_FRONTEND=noninteractive apt full-upgrade -y

echo "=== Base tools ==="
sudo DEBIAN_FRONTEND=noninteractive apt install -y   build-essential dkms git curl unzip time   vulkan-tools libvulkan1 ffmpeg

echo "=== NVIDIA driver candidates ==="
sudo DEBIAN_FRONTEND=noninteractive apt install -y ubuntu-drivers-common
ubuntu-drivers list | tee "$LOG_DIR/ubuntu-drivers-list.txt"

echo "=== Install Ubuntu-recommended NVIDIA driver ==="
sudo ubuntu-drivers install

echo "=== Package snapshot ==="
dpkg-query -W   'nvidia-driver-*' 'nvidia-utils-*' 'linux-image-*' 'linux-headers-*'   > "$LOG_DIR/packages.txt" 2>&1 || true

echo "=== Host bootstrap complete ==="
echo "A reboot inside Ubuntu is now REQUIRED before RoboTwin setup."
echo "Run: sudo reboot"
