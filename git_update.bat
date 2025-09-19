@echo off
REM === Git one-click updater for Windows ===
setlocal ENABLEDELAYEDEXPANSION

REM 1) Lấy thông điệp commit từ tham số, nếu trống thì dùng mặc định
set "MSG=%*"
if "%MSG%"=="" set "MSG=update: chỉnh sửa code"

REM 2) Đảm bảo Git dùng UTF-8 (chạy 1 lần là đủ)
git config i18n.commitencoding utf-8 >NUL 2>&1
git config i18n.logoutputencoding utf-8 >NUL 2>&1
git config core.autocrlf true >NUL 2>&1

echo.
echo ==== GIT STATUS (trước khi add) ====
git status
echo.

REM 3) Add toàn bộ thay đổi (trừ file đã ignore)
git add -A

REM 4) Commit (nếu có thay đổi)
git diff --cached --quiet
if %ERRORLEVEL%==0 (
  echo Không có thay đổi nào để commit. Bỏ qua commit.
) else (
  git commit -m "%MSG%"
  if ERRORLEVEL 1 (
    echo Lỗi khi commit. Dừng lại.
    exit /b 1
  )
)

REM 5) Pull với rebase (đồng bộ với remote)
git pull --rebase origin main
if ERRORLEVEL 1 (
  echo Lỗi khi pull --rebase. Xử lý xung đột rồi chạy lại.
  exit /b 1
)

REM 6) Push lên GitHub
git push origin main
if ERRORLEVEL 1 (
  echo Push thất bại. Kiểm tra kết nối hoặc quyền truy cập.
  exit /b 1
)

echo.
echo ==== HOÀN TẤT ====
git status
endlocal
