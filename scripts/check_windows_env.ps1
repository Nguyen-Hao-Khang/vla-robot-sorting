$ErrorActionPreference = "Continue"

Write-Output "=== Git ==="
git --version

Write-Output "=== Python / uv ==="
python --version
uv --version

Write-Output "=== NVIDIA ==="
nvidia-smi

Write-Output "=== WSL ==="
wsl -l -v

Write-Output "=== Disk ==="
Get-Volume | Select-Object DriveLetter, FileSystemLabel, FileSystem, Size, SizeRemaining, HealthStatus | Format-Table -AutoSize
