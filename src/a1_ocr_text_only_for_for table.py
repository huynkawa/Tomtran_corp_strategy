# -*- coding: utf-8 -*-
"""
clean10_ocr_pipeline.py — OCR thô (PPStructure + Tesseract) cho PDF scan.

Outputs (mirror thư mục input):
  outputs/orc_raw_output/<mirror>/<base>_page{n}.xlsx   (nếu PPStructure ra bảng tốt)
  outputs/orc_raw_output/<mirror>/<base>_page{n}.csv    (fallback Tesseract: schema 5 cột)
  outputs/orc_raw_output/<mirror>/<base>_page{n}_excel.txt
  outputs/orc_raw_output/<mirror>/<base>_page{n}_text.txt
  outputs/orc_raw_output/<mirror>/<base>_page{n}_meta.json
"""

import os, re, cv2, glob, json, shutil, argparse
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple
from pdf2image import convert_from_path
from paddleocr import PPStructure, save_structure_res
import pytesseract
from sklearn.cluster import KMeans

# ====== Cấu hình ======
INPUT_FILE_DEFAULT = None
INPUT_DIR_DEFAULT  = r"inputs/a_text_only_inputs"
OUTPUT_DIR_DEFAULT = r"outputs/a_text_only_outputs"
DPI_DEFAULT        = 500           # DPI cao hơn để nét hơn
MIN_ROWS_DEFAULT   = 5             # Excel < ngưỡng coi như fail và fallback
POPPLER_PATH       = os.environ.get("POPPLER_PATH", None)
TESSERACT_CMD      = os.environ.get("TESSERACT_CMD", None)
if TESSERACT_CMD and os.path.isfile(TESSERACT_CMD):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# ====== Utils ======
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def detect_unit(text: str) -> Optional[str]:
    for p in (r"Đơn vị\s*[:\-]\s*(.+)", r"Don vi\s*[:\-]\s*(.+)", r"Unit\s*[:\-]\s*(.+)"):
        m = re.search(p, text, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip().splitlines()[0]
    return None

def excel_to_text(excel_path: str, source: str) -> str:
    try:
        df = pd.read_excel(excel_path, header=None, engine="openpyxl")
        lines = []
        for row in df.itertuples(index=False):
            cells = [str(c).strip() for c in row if pd.notna(c)]
            row_text = " | ".join([c for c in cells if c])
            if row_text:
                lines.append(row_text)
        table_text = "\n".join(lines)
        return f"[TABLE from {source}]\n{table_text}"
    except Exception as e:
        print(f"⚠️ Không đọc được Excel {excel_path}: {e}")
        return ""

# ====== Chuẩn hoá bảng ======
EXPECTED_COLS = ["Mã số", "Khoản mục/Tài sản", "Thuyết minh", "Số cuối năm", "Số đầu năm"]

def _is_number_like(s: str) -> bool:
    return bool(re.match(r"^[+-]?\d[\d\.,]*$", str(s).strip()))

SECTION_RE = re.compile(r"^(?:[A-Z]\.?|[IVXLC]+\.)$")   # A. | B. | I. | II. | III.
NOTE_SHORT_NUM = re.compile(r"^\d{1,3}$")               # 1..3 -> có thể là số thuyết minh

def _is_number_like(s: str) -> bool:
    return bool(re.match(r"^[+-]?\d[\d\.,]*$", str(s).strip()))

CODE_RE = re.compile(r"^\d{3}(?:\.\d+)?$")  # 100, 131.1, 151.2
NOTE_SHORT_NUM = re.compile(r"^\d{1,3}$")
SECTION_RE = re.compile(r"^(?:[A-Z]\.?|[IVXLC]+\.)$")   # A., B., I., II., ...

def _clean_token(s: str) -> str:
    if s is None:
        return ""
    t = str(s)
    # bỏ ký tự kẻ bảng rơi vào text
    t = t.replace("|", " ").replace("¦", " ").replace("·", " ")
    # rút gọn khoảng trắng
    t = re.sub(r"\s+", " ", t).strip()
    return t

def _is_number_like(s: str) -> bool:
    return bool(re.match(r"^[+-]?\d[\d\.,]*$", str(s).strip()))

def auto_postprocess_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Chuẩn hoá bảng về schema 5 cột:
    - Làm sạch token (bỏ '|')
    - Nếu 'Mã số' không khớp CODE_RE -> đẩy sang 'Khoản mục/Tài sản'
    - Ghép dòng wrap: (code rỗng/không hợp lệ) & không có số liệu
    - Đẩy 'item' toàn số sang 'Số cuối năm' (trừ số thuyết minh ngắn)
    - Chuẩn hoá số về numeric
    """
    EXPECTED_COLS = ["Mã số","Khoản mục/Tài sản","Thuyết minh","Số cuối năm","Số đầu năm"]
    if df is None or df.empty:
        return pd.DataFrame(columns=EXPECTED_COLS)

    # ép đúng 5 cột
    df = df.reset_index(drop=True).copy()
    if df.shape[1] < 5:
        for _ in range(df.shape[1], 5):
            df[df.shape[1]] = ""
    elif df.shape[1] > 5:
        df = df.iloc[:, :5]
    df.columns = EXPECTED_COLS

    # làm sạch token
    for c in df.columns:
        df[c] = df[c].map(_clean_token)

    rows = []
    for _, r in df.iterrows():
        code, item, note, endv, startv = r.tolist()

        # 'IL' -> 'II'
        if note.upper() == "IL":
            note = "II"

        # nếu 'Mã số' KHÔNG hợp lệ -> đẩy sang 'Khoản mục/Tài sản'
        if code and not CODE_RE.fullmatch(code):
            item = (code + " " + item).strip()
            code = ""

        # dòng wrap: không mã số (hoặc mã không hợp lệ), không có số liệu
        if (not code) and (not endv) and (not startv) and rows:
            rows[-1][1] = (rows[-1][1] + " " + item).strip()
            if note:
                rows[-1][2] = (rows[-1][2] + " " + note).strip()
            continue

        # 'item' toàn số -> có thể là số liệu
        if _is_number_like(item) and not NOTE_SHORT_NUM.fullmatch(item) and not endv and not startv:
            endv, item = item, ""

        rows.append([code, item, note, endv, startv])

    out = pd.DataFrame(rows, columns=EXPECTED_COLS)

    # chuẩn hoá số liệu
    for col in ["Số cuối năm","Số đầu năm"]:
        out[col] = (
            out[col].astype(str)
            .str.replace(r"[^\d\-,]", "", regex=True)
            .replace("", None)
        )
        out[col] = pd.to_numeric(out[col].str.replace(",", ""), errors="coerce")

    return out



def check_subtotals(df: pd.DataFrame, tol: int = 5) -> pd.DataFrame:
    """Đánh dấu lệch subtotal (nếu cần)."""
    if df.empty:
        return df
    df = df.copy()
    df["Kiểm tra"] = ""
    # (giữ stub, tuỳ dự án có thể bổ sung công thức sum theo các mục con)
    return df

# --------- Ảnh: deskew nhẹ + tiền xử lý borderless ----------
def _deskew_binary(gray: np.ndarray) -> np.ndarray:
    bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    bw_inv = 255 - bw
    coords = np.column_stack(np.where(bw_inv > 0))
    if coords.size == 0:
        return gray
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    if abs(angle) < 0.2:
        return gray
    (h, w) = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def preprocess_for_borderless(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = _deskew_binary(gray)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enh = clahe.apply(gray)
    thr = cv2.threshold(enh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return cv2.cvtColor(thr, cv2.COLOR_GRAY2BGR)

def detect_table_roi(img_bgr: np.ndarray) -> np.ndarray:
    """
    Tìm ROI bảng lớn nhất bằng morphology đường kẻ.
    Nếu không tìm được, trả về ảnh gốc.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    thr  = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    h, w = thr.shape

    vert = cv2.erode(thr, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, h//60))), 1)
    vert = cv2.dilate(vert, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, h//60))), 2)

    hori = cv2.erode(thr, cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, w//60), 1)), 1)
    hori = cv2.dilate(hori, cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, w//60), 1)), 2)

    grid = cv2.bitwise_or(vert, hori)
    contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img_bgr

    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)

    pad = 6
    y0 = max(0, y - pad); x0 = max(0, x - pad)
    y1 = min(img_bgr.shape[0], y + h + pad); x1 = min(img_bgr.shape[1], x + w + pad)
    return img_bgr[y0:y1, x0:x1]

def read_pdf_pages(pdf_path: str, start_page: int, end_page: Optional[int], dpi: int):
    kwargs = dict(dpi=dpi, first_page=start_page, last_page=end_page)
    if POPPLER_PATH: kwargs["poppler_path"] = POPPLER_PATH
    pages = convert_from_path(pdf_path, **kwargs)
    end_used = end_page if end_page is not None else (start_page + len(pages) - 1)
    return pages, start_page, end_used

def export_excel_from_ppstructure_result(_result, to_path: str) -> bool:
    ok = False
    try:
        tables = []
        for el in _result or []:
            if isinstance(el, dict) and el.get("type") == "table":
                res = el.get("res") or {}
                html = res.get("html")
                if html:
                    try:
                        dfs = pd.read_html(html)
                        if dfs: tables.append(dfs[0])
                    except Exception as ex:
                        print(f"⚠️ read_html fail: {ex}")
        if tables:
            out_df = pd.concat(tables, ignore_index=True)
            out_df = auto_postprocess_table(out_df)
            with pd.ExcelWriter(to_path, engine="xlsxwriter") as writer:
                out_df.to_excel(writer, index=False, sheet_name="Sheet1")
                wb = writer.book; ws = writer.sheets["Sheet1"]
                num_fmt = wb.add_format({"num_format": "#,##0"})
                ws.set_column("A:A", 9)
                ws.set_column("B:B", 60)
                ws.set_column("C:C", 12)
                ws.set_column("D:E", 22, num_fmt)
            ok = True
    except Exception as ex:
        print(f"⚠️ export_excel_from_ppstructure_result fail: {ex}")
    return ok

# --- helper regex dùng cho fallback phân cột ---
CODE_RE = re.compile(r"^\d{3}(?:\.\d+)?$")           # 100 | 131.1 | 151.2
NOTE_RE = re.compile(r"^(?:[IVXLC]+|\d(?:\.\d+)*)$") # I | II | 1 | 1.1 | 1.2.3
NUM_RE  = re.compile(r"^[+-]?\d[\d.,]*$")            # 3.677.178.873.830 | 1,144,179,317,060

def _is_code_token(s: str) -> bool:
    return bool(CODE_RE.fullmatch(str(s).strip()))

def _is_note_token(s: str) -> bool:
    t = str(s).strip().upper()
    # chấp nhận I, II, III, 1, 1.1, 1.2.3 nhưng loại trừ các chuỗi có 3+ chữ số liên tiếp (dễ là tiền)
    if re.fullmatch(r"(?:[IVXLC]+|\d(?:\.\d+){0,2})", t):
        # nếu là số thuần ngắn 1..3 chữ số thì chỉ coi là note nếu nó đứng gần cột "Thuyết minh"
        return True
    return False


def _is_numeric_token(s: str) -> bool:
    return bool(NUM_RE.fullmatch(str(s).strip()))

def _find_header_anchors(df_words: pd.DataFrame) -> dict | None:
    """
    Tìm vị trí x (left) của các cột từ dòng header: 'Mã số', 'Thuyết minh', 'Số cuối năm', 'Số đầu năm'.
    Trả về dict: {'code': x_code_right_bound, 'note': x_note, 'end': x_end, 'start': x_start}
    Nếu không thấy đủ, trả về None.
    """
    if df_words.empty:
        return None

    def norm(s: str) -> str:
        t = str(s).lower().strip()
        rep = {
            "â":"a","ă":"a","á":"a","à":"a","ả":"a","ã":"a","ạ":"a",
            "ê":"e","é":"e","è":"e","ẻ":"e","ẽ":"e","ẹ":"e",
            "ô":"o","ơ":"o","ó":"o","ò":"o","ỏ":"o","õ":"o","ọ":"o",
            "ư":"u","ú":"u","ù":"u","ủ":"u","ũ":"u","ụ":"u",
            "í":"i","ì":"i","ỉ":"i","ĩ":"i","ị":"i",
            "ý":"y","ỳ":"y","ỷ":"y","ỹ":"y","ỵ":"y","đ":"d"
        }
        for k,v in rep.items():
            t = t.replace(k, v)
        return t

    words = df_words.copy()
    words["norm"] = words["text"].map(norm)

    cand_code  = words[words["norm"].str.contains(r"\bma\s*so\b")]
    cand_note  = words[words["norm"].str.contains(r"\bthuyet\s*minh\b")]
    cand_end   = words[words["norm"].str.contains(r"\bso\s*cuoi\b|\bso\s*cuoi\s*nam\b")]
    cand_start = words[words["norm"].str.contains(r"\bso\s*dau\b|\bso\s*dau\s*nam\b")]

    if cand_note.empty or cand_end.empty or cand_start.empty:
        return None

    x_note  = float(cand_note.sort_values("left").iloc[0]["left"])
    x_end   = float(cand_end.sort_values("left").iloc[0]["left"])
    x_start = float(cand_start.sort_values("left").iloc[0]["left"])

    left_min = float(words["left"].min())
    code_right_bound = left_min + (x_note - left_min) * 0.30

    return {"code": code_right_bound, "note": x_note, "end": x_end, "start": x_start}


def _group_lines(df_words: pd.DataFrame) -> list[pd.DataFrame]:
    """Nhóm token theo dòng với ngưỡng động từ median height."""
    lines, cur, last_y = [], [], None
    if df_words.empty: return lines
    thr = max(8.0, float(df_words["height"].median()) * 0.9)
    for _, r in df_words.sort_values(["y_center", "left"]).iterrows():
        y = r["y_center"]
        if last_y is None or abs(y - last_y) <= thr:
            cur.append(r)
        else:
            lines.append(pd.DataFrame(cur)); cur = [r]
        last_y = y
    if cur: lines.append(pd.DataFrame(cur))
    return lines

def tesseract_csv_fallback(page_bgr: np.ndarray, csv_path: str):
    """
    Fallback chính xác hơn:
    - Crop ROI bảng
    - Lấy tâm 2 cột số từ token số; tâm 1 cột note từ token note
    - Assign cột dựa trên nội dung + gần tâm
    - Ghép dòng theo ngưỡng động
    - Bỏ phần rác đến trước dòng 'mã số' đầu tiên
    - Xuất CSV + Excel (định dạng số, set width)
    """
    try:
        from pytesseract import Output

        roi = detect_table_roi(page_bgr)

        data = pytesseract.image_to_data(roi, lang="eng+vie", output_type=Output.DATAFRAME)
        data = data.dropna(subset=["text"])
        data = data[data["text"].astype(str).str.strip().astype(bool)]
        if data.empty:
            pd.DataFrame(columns=EXPECTED_COLS).to_csv(csv_path, index=False, encoding="utf-8-sig")
            print(f"📄 Fallback CSV (trống): {csv_path}")
            return

        # toạ độ trung tâm dọc
        data["y_center"] = data["top"] + data["height"] / 2

        # --- 1) Ước lượng tâm cột (ưu tiên neo header nếu có) ---
        anchors = _find_header_anchors(data)

        if anchors:
            center_note   = anchors["note"]
            center_end    = anchors["end"]
            center_start  = anchors["start"]
            code_right_bound = anchors["code"]
        else:
            nums = data[data["text"].apply(_is_numeric_token)]
            if len(nums) >= 4:
                km2 = KMeans(n_clusters=2, n_init=10, random_state=0).fit(nums[["left"]].to_numpy())
                centers_num = sorted(list(km2.cluster_centers_.ravel()))
            else:
                xs = sorted(data["left"].tolist())
                centers_num = [np.percentile(xs, 70), np.percentile(xs, 90)]

            notes = data[data["text"].apply(_is_note_token)]
            if len(notes) >= 3:
                km1 = KMeans(n_clusters=1, n_init=5, random_state=0).fit(notes[["left"]].to_numpy())
                center_note = float(km1.cluster_centers_.ravel()[0])
            else:
                center_note = float(np.percentile(data["left"], 55))

            center_end, center_start = centers_num[0], centers_num[1]  # trái hơn = 'Số cuối năm'
            left_min = float(np.percentile(data["left"], 5))
            code_right_bound = left_min + (center_note - left_min) * 0.30

    
        # mở rộng ranh giới để tránh đẩy nhầm
        # đẩy ranh giới phải của "Mã số" ra 45% khoảng cách tới cột "Thuyết minh"
        # thu hẹp cột "Mã số" để chữ như "Các", "Tài" không rơi vào A
        code_right_bound = left_min + (center_note - left_min) * 0.22   # 22% bề ngang trái
        item_right_bound = center_note - (center_note - code_right_bound) * 0.35
        note_right_bound = center_note + (center_end - center_note) * 0.30


        # --- 2) Gom dòng ---
        lines = _group_lines(data)

        out_rows = []
        for g in lines:
            row = [""] * 5
            for _, tok in g.sort_values("left").iterrows():
                t  = str(tok["text"]).strip()
                lx = float(tok["left"])

                # Ưu tiên NGỮ NGHĨA trước
                if _is_code_token(t) and lx <= code_right_bound:
                    # đã có code mà lại gặp I., 1.1... -> đưa vào thuyết minh
                    if row[0] and _is_note_token(t):
                        row[2] = (row[2] + " " + t).strip()
                    else:
                        row[0] = (row[0] + " " + t).strip()
                    continue

                # Tiêu đề/mục lớn (I., II., A., B., ...)
                if SECTION_RE.fullmatch(t.replace(" ", "")) and lx < center_note:
                    row[1] = (row[1] + " " + t).strip()
                    continue

                if _is_note_token(t) and lx <= note_right_bound and lx >= (center_note - (center_end - center_note)*0.8):
                    row[2] = (row[2] + " " + t).strip()
                    continue

                # cứu cánh: '5.2', '6', '7' hơi lệch phải
                if _is_note_token(t) and lx < (center_end - (center_end - center_note)*0.35):
                    row[2] = (row[2] + " " + t).strip()
                    continue

                if _is_numeric_token(t):
                    # chọn cột số gần tâm nào hơn
                    dest = 3 if abs(lx - center_end) <= abs(lx - center_start) else 4
                    row[dest] = (row[dest] + " " + t).strip()
                    continue

                # Không khớp ngữ nghĩa -> dùng vị trí
                if lx <= code_right_bound:
                    row[0] = (row[0] + " " + t).strip()
                elif lx <= item_right_bound:
                    row[1] = (row[1] + " " + t).strip()
                elif lx <= note_right_bound:
                    row[2] = (row[2] + " " + t).strip()
                else:
                    row[3] = (row[3] + " " + t).strip()

            out_rows.append(row)


        df = pd.DataFrame(out_rows, columns=EXPECTED_COLS)

        # --- 3) Bỏ phần rác trước dòng mã số đầu tiên ---
        first_idx = None
        for i, v in enumerate(df["Mã số"]):
            if _is_code_token(v):
                first_idx = i; break
        if first_idx is not None:
            df = df.iloc[first_idx:].reset_index(drop=True)

        # --- 4) Chuẩn hoá & ghi file ---
        df = auto_postprocess_table(df)
        df = check_subtotals(df)

        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        excel_path = csv_path.replace(".csv", ".xlsx")
        with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False, sheet_name="Sheet1")
            wb = writer.book; ws = writer.sheets["Sheet1"]
            num_fmt = wb.add_format({"num_format": "#,##0"})
            ws.set_column("A:A", 9)
            ws.set_column("B:B", 60)
            ws.set_column("C:C", 12)
            ws.set_column("D:E", 22, num_fmt)

        print(f"📄 Fallback CSV + Excel (improved): {csv_path}, {excel_path}")
    except Exception as e:
        print(f"⚠️ Fallback CSV lỗi: {e}")


# ====== Core ======
def process_pdf(pdf_path: str, input_root: str, out_root: str,
                start_page: int, end_page: Optional[int], dpi: int,
                use_preprocess: bool, min_rows_excel: int):
    print(f"\n📂 Đang xử lý file: {pdf_path}")

    rel_path = os.path.relpath(pdf_path, input_root)
    out_dir = os.path.join(out_root, os.path.dirname(rel_path))
    ensure_dir(out_dir)

    base = os.path.splitext(os.path.basename(pdf_path))[0]

    table_engine = PPStructure(show_log=False, use_gpu=False, layout=True, recovery=True)

    pages, s_used, e_used = read_pdf_pages(pdf_path, start_page, end_page, dpi)
    print(f"📑 Xử lý trang {s_used} → {e_used} (DPI={dpi})")

    current_unit: Optional[str] = None

    for pno, pil_img in enumerate(pages, start=s_used):
        bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        roi_bgr = detect_table_roi(bgr)
        img_for_table = preprocess_for_borderless(roi_bgr) if use_preprocess else roi_bgr

        excel_file = os.path.join(out_dir, f"{base}_page{pno}.xlsx")
        csv_file   = os.path.join(out_dir, f"{base}_page{pno}.csv")

        # 1) PPStructure trước
        moved_excel = False
        try:
            result = table_engine(img_for_table)
            tables = []
            for el in result or []:
                if isinstance(el, dict) and el.get("type") == "table":
                    html = (el.get("res") or {}).get("html")
                    if html:
                        try:
                            dfs = pd.read_html(html)
                            if dfs: tables.append(dfs[0])
                        except Exception as ex:
                            print(f"⚠️ read_html fail: {ex}")
            if tables:
                df = pd.concat(tables, ignore_index=True)
                df = auto_postprocess_table(df)
                with pd.ExcelWriter(excel_file, engine="xlsxwriter") as writer:
                    df.to_excel(writer, index=False, sheet_name="Sheet1")
                    wb = writer.book; ws = writer.sheets["Sheet1"]
                    num_fmt = wb.add_format({"num_format": "#,##0"})
                    ws.set_column("A:A", 9)
                    ws.set_column("B:B", 60)
                    ws.set_column("C:C", 12)
                    ws.set_column("D:E", 22, num_fmt)
                df.to_csv(csv_file, index=False, encoding="utf-8-sig")
                moved_excel = True
        except Exception as e:
            print(f"⚠️ PPStructure lỗi trang {pno}: {e}")

        # 2) Fallback nếu không có Excel
        if not moved_excel:
            tesseract_csv_fallback(roi_bgr, csv_file)

        # 3) Nếu đã có Excel → kiểm tra số dòng & xuất text đối chiếu
        if os.path.isfile(excel_file):
            try:
                df_check = pd.read_excel(excel_file, engine="openpyxl")
                df_check = auto_postprocess_table(df_check)
                with pd.ExcelWriter(excel_file, engine="xlsxwriter") as writer:
                    df_check.to_excel(writer, index=False, sheet_name="Sheet1")
                    wb = writer.book; ws = writer.sheets["Sheet1"]
                    num_fmt = wb.add_format({"num_format": "#,##0"})
                    ws.set_column("A:A", 9)
                    ws.set_column("B:B", 60)
                    ws.set_column("C:C", 12)
                    ws.set_column("D:E", 22, num_fmt)
                df_check.to_csv(csv_file, index=False, encoding="utf-8-sig")

                if df_check.shape[0] < min_rows_excel:
                    print(f"⚠️ Excel {excel_file} chỉ có {df_check.shape[0]} dòng (<{min_rows_excel}) → fallback CSV.")
                    try:
                        os.remove(excel_file)
                    except Exception as e:
                        print(f"⚠️ Không xoá được {excel_file}: {e}")
                    tesseract_csv_fallback(roi_bgr, csv_file)
                else:
                    txt = excel_to_text(excel_file, f"{base}_page{pno}")
                    if txt:
                        with open(excel_file.replace(".xlsx", "_excel.txt"), "w", encoding="utf-8") as f:
                            f.write(txt)
            except Exception as e:
                print(f"⚠️ Lỗi Excel {excel_file}: {e}")
                tesseract_csv_fallback(roi_bgr, csv_file)

        # 4) Text toàn trang (không crop) + metadata
        try:
            txt_tess = pytesseract.image_to_string(bgr, lang="eng+vie")
        except pytesseract.TesseractNotFoundError:
            print("⚠️ Không tìm thấy Tesseract. Set TESSERACT_CMD tới tesseract.exe.")
            txt_tess = ""
        with open(os.path.join(out_dir, f"{base}_page{pno}_text.txt"), "w", encoding="utf-8") as f:
            f.write(txt_tess)

        unit_found = detect_unit(txt_tess)
        if unit_found:
            current_unit = unit_found

        meta = {
            "file": base, "page": pno, "unit": current_unit,
            "source_pdf": os.path.abspath(pdf_path), "dpi": dpi,
            "preprocess": use_preprocess,
        }
        with open(os.path.join(out_dir, f"{base}_page{pno}_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

# ====== Runner ======
def run_ocr_pipeline(input_file: Optional[str] = INPUT_FILE_DEFAULT,
                     input_dir: str = INPUT_DIR_DEFAULT,
                     out_dir: str = OUTPUT_DIR_DEFAULT,
                     start_page: int = 1,
                     end_page: Optional[int] = None,
                     dpi: int = DPI_DEFAULT,
                     use_preprocess: bool = True,
                     force_clean_out: bool = True,
                     min_rows_excel: int = MIN_ROWS_DEFAULT):
    # Hỏi người dùng nếu output đã tồn tại
    if os.path.exists(out_dir):
        choice = input(f"⚠️ Output {out_dir} đã tồn tại. "
                       "Chọn y=xoá build lại, a=append thêm file mới, n=bỏ qua: ").strip().lower()
        if choice == "y":
            shutil.rmtree(out_dir, ignore_errors=True); print(f"🗑️ Đã xoá {out_dir}")
        elif choice == "n":
            print("⏭️ Bỏ qua OCR pipeline."); return
        elif choice == "a":
            print(f"➕ Giữ {out_dir}, OCR thêm file mới.")
        else:
            print("❌ Lựa chọn không hợp lệ, bỏ qua."); return
    ensure_dir(out_dir)

    if input_file:
        if not os.path.isfile(input_file):
            print(f"⚠️ Không tìm thấy file: {input_file}"); return
        process_pdf(input_file,
                    os.path.dirname(input_file) if os.path.isdir(input_dir) else input_dir,
                    out_dir, start_page, end_page, dpi, use_preprocess, min_rows_excel)
    else:
        if not os.path.isdir(input_dir):
            print(f"⚠️ Không tìm thấy thư mục: {input_dir}"); return
        pdf_files = glob.glob(os.path.join(input_dir, "**", "*.pdf"), recursive=True)
        if not pdf_files:
            print("⚠️ Không tìm thấy PDF nào."); return
        for pdf in pdf_files:
            process_pdf(pdf, input_dir, out_dir, start_page, end_page, dpi, use_preprocess, min_rows_excel)

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("OCR pipeline (PPStructure + Tesseract) cho PDF scan")
    src = p.add_mutually_exclusive_group(required=False)
    src.add_argument("--file", type=str, help="Đường dẫn PDF cần OCR")
    src.add_argument("--dir", type=str, help="Thư mục chứa PDF (recursive)")
    p.add_argument("--out", type=str, default=OUTPUT_DIR_DEFAULT, help="Thư mục output")
    p.add_argument("--start", type=int, default=1, help="Trang bắt đầu (1-based)")
    p.add_argument("--end", type=int, default=None, help="Trang kết thúc (1-based, inclusive)")
    p.add_argument("--dpi", type=int, default=DPI_DEFAULT, help="DPI render ảnh từ PDF")
    p.add_argument("--no-pre", action="store_true", help="Tắt tiền xử lý ảnh (mặc định bật)")
    p.add_argument("--keep-out", action="store_true", help="Không xoá output cũ (mặc định xoá)")
    p.add_argument("--min-rows", type=int, default=MIN_ROWS_DEFAULT, help="Excel < min-rows → fallback CSV")
    return p

if __name__ == "__main__":
    args = build_argparser().parse_args()
    run_ocr_pipeline(
        input_file=args.file,
        input_dir=args.dir if args.dir else INPUT_DIR_DEFAULT,
        out_dir=args.out,
        start_page=args.start,
        end_page=args.end,
        dpi=args.dpi,
        use_preprocess=(not args.no_pre),
        force_clean_out=(not args.keep_out),
        min_rows_excel=args.min_rows,
    )
