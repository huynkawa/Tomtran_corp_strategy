# -*- coding: utf-8 -*-
"""
src/c1_ocr_scan_to_png_bctc.py — P1: Prelight OCR (deskew + crop margins + binarize)
Chạy trước khi OCR chính để tạo ảnh sạch cho bước sau.

- Hỗ trợ PDF & ảnh (PNG/JPG/TIF)
- Render PDF ở DPI mặc định 500
- Ước lượng & sửa nghiêng (HoughLinesP) nếu |góc| > 0.3°
- Cắt lề trắng, chừa padding
- Binarize nhẹ (adaptive) giữ đường kẻ mảnh
- Xuất "mirror" cấu trúc thư mục, mỗi trang 2 file:
    *_orig.png  (xám, đã deskew + crop)
    *_bin.png   (binarize)
"""

from __future__ import annotations
import os, glob, argparse
from typing import Optional, Tuple, List
import numpy as np
import cv2
import shutil   # [ADD]
APPEND_MODE = False  # [ADD] dùng để biết có đang chạy chế độ append-only không


# pdf2image
try:
    from pdf2image import convert_from_path
except Exception:
    convert_from_path = None

# ========= ĐƯỜNG DẪN MẶC ĐỊNH (Windows) =========
DEFAULT_IN  = r"D:\1.TLAT\3. ChatBot_project\1_Insurance_Strategy\inputs\c_scan_inputs_test"
DEFAULT_OUT = r"D:\1.TLAT\3. ChatBot_project\1_Insurance_Strategy\outputs\outputs0\c0_ocr_scan_bctc_to_png"
POPPLER_PATH = os.environ.get("POPPLER_PATH", None)  # set nếu cần

# ========= TIỆN ÍCH =========
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def rel_out_dir(in_root: str, out_root: str, file_path: str) -> str:
    rel = os.path.relpath(os.path.dirname(file_path), in_root)
    return out_root if rel == "." else os.path.join(out_root, rel)

def imwrite_ok(path: str, img) -> None:
    ensure_dir(os.path.dirname(path))
    ok = cv2.imwrite(path, img)
    if not ok:
        raise RuntimeError(f"Không ghi được ảnh: {path}")

def is_pdf(path: str) -> bool:
    return os.path.splitext(path)[1].lower() == ".pdf"

def is_image(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]

# ========= CORE: ƯỚC LƯỢNG NGHIÊNG / CROP / BINARIZE =========
def estimate_skew_angle(gray: np.ndarray) -> float:
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=160,
                            minLineLength=max(gray.shape)//10, maxLineGap=20)
    if lines is None:
        return 0.0
    angles = []
    for l in lines:
        x1, y1, x2, y2 = l[0]
        ang = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        while ang >= 90: ang -= 180
        while ang <  -90: ang += 180
        angles.append(ang)
    if not angles:
        return 0.0
    a = np.array(angles)
    near_h = a[np.abs(a) <= 30]
    near_v = a[(np.abs(np.abs(a) - 90) <= 30)]
    if len(near_h) >= len(near_v):
        return float(np.median(near_h)) if len(near_h) else 0.0
    deltas = []
    for v in near_v:
        deltas.append(v - 90 if v > 0 else v + 90)
    return float(np.median(deltas)) if deltas else 0.0

def rotate_bound(img: np.ndarray, angle_deg: float) -> np.ndarray:
    (h, w) = img.shape[:2]
    c = (w//2, h//2)
    M = cv2.getRotationMatrix2D(c, angle_deg, 1.0)
    cos, sin = abs(M[0,0]), abs(M[0,1])
    nW = int((h*sin) + (w*cos)); nH = int((h*cos) + (w*sin))
    M[0,2] += (nW/2) - c[0]; M[1,2] += (nH/2) - c[1]
    return cv2.warpAffine(img, M, (nW, nH), flags=cv2.INTER_CUBIC, borderValue=(255,255,255))

def crop_margins(gray: np.ndarray, pad: int=16) -> Tuple[np.ndarray, Tuple[int,int,int,int]]:
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    inv = 255 - thr
    ys, xs = np.where(inv > 0)
    if len(xs) == 0 or len(ys) == 0:
        return gray, (0,0,gray.shape[1], gray.shape[0])
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    x0 = max(0, x0 - pad); y0 = max(0, y0 - pad)
    x1 = min(gray.shape[1]-1, x1 + pad)
    y1 = min(gray.shape[0]-1, y1 + pad)
    return gray[y0:y1+1, x0:x1+1], (x0,y0,x1-x0+1,y1-y0+1)

def gentle_binarize(gray: np.ndarray) -> np.ndarray:
    blur = cv2.medianBlur(gray, 3)
    return cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 31, 15)

# ========= XỬ LÝ ẢNH / PDF =========
def process_image(in_root: str, out_root: str, img_path: str, base: str, page_no: int) -> None:
    out_dir = rel_out_dir(in_root, out_root, img_path)
    ensure_dir(out_dir)

    data = np.fromfile(img_path, dtype=np.uint8)
    im = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if im is None:
        print(f"⚠️ Bỏ qua (không đọc được): {img_path}")
        return

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    angle = estimate_skew_angle(gray)
    if abs(angle) > 0.3:
        im = rotate_bound(im, -angle)
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # ảnh xám đã crop (dùng cho _orig.png)
    cropped, _ = crop_margins(gray, pad=16)
    # ảnh nhị phân (dùng cho _bin.png)
    bin_img = gentle_binarize(cropped)

    suf = f"{base}_page{page_no:02d}"
    p_orig = os.path.join(out_dir, f"{suf}_orig.png")
    p_bin  = os.path.join(out_dir, f"{suf}_bin.png")

    # ===== ghi file với chế độ append-only =====
    # nếu APPEND_MODE=True và file đã tồn tại → bỏ qua, ngược lại ghi bình thường
    if APPEND_MODE and os.path.exists(p_orig):
        print(f"↩️ Bỏ qua (đã có): {os.path.relpath(p_orig, out_root)}")
    else:
        imwrite_ok(p_orig, cv2.cvtColor(cropped, cv2.COLOR_GRAY2BGR))

    if APPEND_MODE and os.path.exists(p_bin):
        print(f"↩️ Bỏ qua (đã có): {os.path.relpath(p_bin, out_root)}")
    else:
        imwrite_ok(p_bin,  cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR))

    print(f"🖼️ Saved: {os.path.relpath(p_orig, out_root)}, {os.path.relpath(p_bin, out_root)}  (skew={angle:.2f}°)")


def process_pdf(in_root: str, out_root: str, pdf_path: str, dpi: int, start: int, end: Optional[int]) -> None:
    if convert_from_path is None:
        raise RuntimeError("Thiếu pdf2image. pip install pdf2image và thiết lập POPPLER_PATH nếu cần.")
    kwargs = dict(dpi=dpi, first_page=start, last_page=end)
    if POPPLER_PATH: kwargs["poppler_path"] = POPPLER_PATH
    try:
        pages = convert_from_path(pdf_path, **kwargs)
    except Exception as e:
        print(f"⚠️ Lỗi render PDF '{pdf_path}': {e}"); return
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    out_dir = rel_out_dir(in_root, out_root, pdf_path)
    ensure_dir(out_dir)
    for i, pil in enumerate(pages, start=start):
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        angle = estimate_skew_angle(gray)
        if abs(angle) > 0.3:
            img = rotate_bound(img, -angle); gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cropped, _ = crop_margins(gray, pad=16)
        bin_img = gentle_binarize(cropped)
        suf = f"{base}_page{i:02d}"
        p_orig = os.path.join(out_dir, f"{suf}_orig.png")
        p_bin  = os.path.join(out_dir, f"{suf}_bin.png")
        imwrite_ok(p_orig, cv2.cvtColor(cropped, cv2.COLOR_GRAY2BGR))
        imwrite_ok(p_bin,  cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR))
        print(f"🖼️ Saved: {os.path.relpath(p_orig, out_root)}, {os.path.relpath(p_bin, out_root)}  (skew={angle:.2f}°)")

def walk_and_process(in_root: str, out_root: str, dpi: int, start: int, end: Optional[int]) -> None:
    pdfs = glob.glob(os.path.join(in_root, "**", "*.pdf"), recursive=True)
    imgs: List[str] = []
    for pat in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"):
        imgs.extend(glob.glob(os.path.join(in_root, "**", pat), recursive=True))
    if not pdfs and not imgs:
        print("⚠️ Không tìm thấy PDF/Ảnh nào trong input."); return
    for p in pdfs:
        print(f"\n📂 PDF: {p}"); process_pdf(in_root, out_root, p, dpi, start, end)
    for p in imgs:
        print(f"\n🖼️ Ảnh: {p}"); base = os.path.splitext(os.path.basename(p))[0]
        process_image(in_root, out_root, p, base, page_no=1)

# ========= CLI =========
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("P1 Prelight OCR (deskew + crop + binarize)")
    p.add_argument("--in",   dest="in_dir",  default=DEFAULT_IN,
                   help="Thư mục input (recursive) chứa PDF/ảnh")
    p.add_argument("--out",  dest="out_dir", default=DEFAULT_OUT,
                   help="Thư mục output mirror")
    p.add_argument("--dpi", type=int, default=500, help="DPI render PDF")
    p.add_argument("--start", type=int, default=1, help="Trang bắt đầu (PDF)")
    p.add_argument("--end",   type=int, default=None, help="Trang kết thúc (PDF) (inclusive)")

    # [ADD] hỏi/xoá/append/bỏ qua khi output đã tồn tại
    p.add_argument("--clean", choices=["ask","y","a","n"], default="ask",
                help="ask: hỏi; y: xoá output cũ; a: giữ thư mục & chỉ ghi file mới; n: bỏ qua nếu đã tồn tại")
    return p


def main():
    global APPEND_MODE  # phải đứng TRƯỚC mọi phép gán APPEND_MODE trong hàm

    args = build_argparser().parse_args()

    # Chuẩn bị output theo --clean
    out_root = args.out_dir
    if os.path.exists(out_root):
        choice = args.clean
        if choice == "ask":
            choice = input(f"⚠️ Output '{out_root}' đã tồn tại. y=xoá, a=append, n=bỏ qua: ").strip().lower()
        if choice == "y":
            shutil.rmtree(out_root, ignore_errors=True); print(f"🗑️ Đã xoá {out_root}")
        elif choice == "n":
            print("⏭️ Bỏ qua P1 (prelight)."); return
        elif choice == "a":
            print(f"➕ Giữ {out_root}, chỉ ghi file mới.")
        else:
            print("❌ Lựa chọn không hợp lệ → bỏ qua."); return

    ensure_dir(out_root)

    # BẬT cờ append sau khi đã khai báo global
    APPEND_MODE = (args.clean == "a")

    print(f"Input : {args.in_dir}")
    print(f"Output: {args.out_dir}")
    print(f"DPI   : {args.dpi}")
    if POPPLER_PATH: print(f"POPPLER_PATH: {POPPLER_PATH}")

    walk_and_process(args.in_dir, args.out_dir, args.dpi, args.start, args.end)
    print("\n✅ Hoàn tất P1 Prelight OCR. Dùng *_bin.png cho bước OCR tiếp theo.")



if __name__ == "__main__":
    main()
