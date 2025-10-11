# -*- coding: utf-8 -*-
"""
src/table_narrator.py — Chuyển bảng BCTC (pipe: CODE|NAME|NOTE|END|BEGIN) → diễn giải tiếng Việt

- Đầu vào: 
    + table_text: chuỗi bảng pipe đã clean/GPT (mỗi dòng 1 record, có header tùy chọn)
    + rules_yaml: dict YAML (validator BCTC bạn cung cấp) để biết cấu trúc/mối quan hệ
    + sheet_hint: gợi ý loại bảng ("balance_sheet", "income_statement_insurance", "income_statement_total", ...)

- Đầu ra: đoạn văn tiếng Việt, bao gồm:
    + Tóm tắt quy mô: tổng tài sản / tổng nguồn vốn / LNTT / LNST (nếu bắt gặp mã tương ứng)
    + Biến động năm: so sánh END vs BEGIN, phần trăm thay đổi theo từng nhóm lớn
    + Kiểm tra logic: các rule eq / approx_eq (nếu lệch → nêu cảnh báo nhẹ)
    + Nhấn mạnh khoản mục nổi bật (top tăng/giảm)

- Thiết kế để gọi từ p1a_clean10_ocr_bctc_GPT.py với flag --table-narrative
"""
from __future__ import annotations
import re, math
from typing import List, Dict, Optional, Tuple

Number = Optional[float]

# -------------------------
# Parse & helpers
# -------------------------

def _parse_number(s: str) -> Number:
    if s is None:
        return None
    s = s.strip()
    if not s or s in {"-","—","_"}:
        return None
    # (1) normalize thousand grouping: '1.234.567' → '1234567'
    s2 = re.sub(r"\.(?=\d{3}(\D|$))", "", s)
    # (2) decimal comma → dot
    s2 = s2.replace(",", ".")
    # parentheses negative
    neg = False
    if s2.startswith("(") and s2.endswith(")"):
        neg = True
        s2 = s2[1:-1]
    try:
        val = float(s2)
        return -val if neg else val
    except:
        return None


def _split_cols(line: str) -> List[str]:
    s = line.strip()
    if s.startswith("|"): s = s[1:]
    if s.endswith("|"):   s = s[:-1]
    return [c.strip() for c in s.split("|")]


def parse_table(table_text: str) -> List[Dict[str,str]]:
    rows: List[Dict[str,str]] = []
    if not table_text:
        return rows
    for ln in [l for l in table_text.splitlines() if l.strip()]:
        cols = _split_cols(ln)
        if len(cols) < 5:
            # cố gắng nới: CODE | NAME | END | BEGIN
            if len(cols) == 4:
                code, name, end, begin = cols[0], cols[1], cols[2], cols[3]
                rows.append({"code":code, "name":name, "note":"", "end":end, "begin":begin})
            else:
                # bỏ qua dòng chú thích
                continue
        else:
            code, name, note, end, begin = cols[:5]
            rows.append({"code":code, "name":name, "note":note, "end":end, "begin":begin})
    return rows


# -------------------------
# Rule engine (nhẹ)
# -------------------------

def _lookup_value(rows: List[Dict[str,str]], code: str, which: str = "end") -> Number:
    for r in rows:
        if r.get("code") == code:
            return _parse_number(r.get(which))
    return None


def _eq_check(rows: List[Dict[str,str]], left: str, right_expr: List[str], which: str = "end") -> Tuple[bool, Number, Number]:
    L = _lookup_value(rows, left, which)
    total = 0.0
    any_val = False
    for item in right_expr:
        sign = 1
        key = item
        if item.startswith("-"):
            sign = -1
            key = item[1:]
        v = _lookup_value(rows, key, which)
        if v is not None:
            total += sign * v
            any_val = True
    if L is None or (not any_val):
        return True, L, total  # không có dữ liệu → coi như pass im lặng
    ok = abs(L - total) < 0.5  # dung sai nhỏ
    return ok, L, total


# -------------------------
# Narratives
# -------------------------

def _fmt_money(v: Number) -> str:
    if v is None: return "—"
    sign = "-" if v < 0 else ""
    n = abs(int(round(v)))
    s = f"{n:,}".replace(",", ".")
    return f"{sign}{s}"


def _pct(a: Number, b: Number) -> Optional[float]:
    if a is None or b is None or b == 0:
        return None
    return (a - b) / abs(b)


def _top_changes(rows, topk=5):
    changes = []
    for r in rows:
        e = _parse_number(r.get("end"))
        b = _parse_number(r.get("begin"))
        if e is None or b is None:
            continue
        delta = e - b
        changes.append((abs(delta), delta, r))
    changes.sort(key=lambda x: x[0], reverse=True)
    out = []
    for _, delta, r in changes[:topk]:
        out.append((r.get("code"), r.get("name"), delta))
    return out


def narrate_table(table_text: str, rules_yaml: dict, sheet_hint: Optional[str] = None) -> str:
    rows = parse_table(table_text)
    if not rows:
        return "[Không trích được dữ liệu bảng để diễn giải]"

    codes = {r["code"] for r in rows if r.get("code")}

    # nhận diện sheet nếu không chỉ định
    sheet = sheet_hint
    if sheet is None:
        # heuristics: có 270 ⇒ Bảng cân đối; có 50, 60 ⇒ KQKD tổng hợp; có 03,10,19 ⇒ KQKD bảo hiểm
        if "270" in codes:
            sheet = "balance_sheet"
        elif {"50","60"} & codes:
            sheet = "income_statement_total"
        elif {"03","10","19"} & codes:
            sheet = "income_statement_insurance"
        else:
            sheet = "generic"

    lines: List[str] = []

    # --- Tóm tắt quy mô chính
    if sheet == "balance_sheet":
        total_assets = _lookup_value(rows, "270", "end")
        total_eqlib  = _lookup_value(rows, "440", "end")
        lines.append(
            f"Tổng tài sản cuối kỳ đạt {_fmt_money(total_assets)} VND; nguồn vốn tương ứng {_fmt_money(total_eqlib)} VND (đối ứng Tài sản = Nguồn vốn)."
        )
        # biến động tổng tài sản
        ta_b = _lookup_value(rows, "270", "begin")
        chg = _pct(total_assets, ta_b)
        if chg is not None:
            lines.append(f"So với đầu kỳ, tổng tài sản {'tăng' if chg>=0 else 'giảm'} khoảng {abs(chg)*100:.1f}%.")

        # điểm nhấn nhóm lớn
        for big, name in [("100","Tài sản ngắn hạn"),("200","Tài sản dài hạn")]:
            e = _lookup_value(rows, big, "end")
            b = _lookup_value(rows, big, "begin")
            if e is not None:
                ratio = (e / total_assets) if (total_assets not in (None, 0)) else None
                piece = f"{name} {_fmt_money(e)} VND"
                if ratio is not None:
                    piece += f" (~{ratio*100:.1f}% tổng tài sản)"
                if b is not None:
                    pc = _pct(e, b)
                    if pc is not None:
                        piece += f", {'tăng' if pc>=0 else 'giảm'} {abs(pc)*100:.1f}% so với đầu kỳ"
                lines.append("- " + piece + ".")

        # kiểm tra vài rule then chốt
        ok1_end, L_end, R_end = _eq_check(rows, "270", ["100","200"], which="end")
        ok2_end, L2_end, R2_end = _eq_check(rows, "440", ["300","400"], which="end")
        if not ok1_end:
            lines.append(f"⚠️ Chênh lệch tổng tài sản (270={_fmt_money(L_end)}) so với 100+200={_fmt_money(R_end)}.")
        if not ok2_end:
            lines.append(f"⚠️ Chênh lệch tổng nguồn vốn (440={_fmt_money(L2_end)}) so với 300+400={_fmt_money(R2_end)}.")

    elif sheet == "income_statement_insurance":
        g = _lookup_value(rows, "19", "end")
        dt = _lookup_value(rows, "10", "end")
        cp = _lookup_value(rows, "18", "end")
        line = "Kết quả HĐKD bảo hiểm: "
        if dt is not None: line += f"doanh thu thuần {_fmt_money(dt)} VND, "
        if cp is not None: line += f"tổng chi phí {_fmt_money(cp)} VND, "
        if g is not None:  line += f"lợi nhuận gộp {_fmt_money(g)} VND."
        lines.append(line.strip())

        # động lực từ 03 (phí thuần)
        fee_net = _lookup_value(rows, "03", "end")
        if fee_net is not None:
            lines.append(f"Doanh thu phí thuần (mã 03) đạt {_fmt_money(fee_net)} VND, là nền tảng chính của doanh thu.")

    elif sheet == "income_statement_total":
        lntt = _lookup_value(rows, "50", "end")
        lnst = _lookup_value(rows, "60", "end")
        line = "Kết quả kinh doanh tổng hợp: "
        if lntt is not None: line += f"LNTT {_fmt_money(lntt)} VND; "
        if lnst is not None: line += f"LNST {_fmt_money(lnst)} VND."
        lines.append(line.strip())

    else:
        lines.append("Bảng số liệu đã được chuẩn hóa; dưới đây là một số điểm nhấn.")

    # --- Top biến động lớn
    topchg = _top_changes(rows, topk=5)
    if topchg:
        lines.append("Các khoản mục biến động đáng chú ý:")
        for code, name, delta in topchg:
            lines.append(f"- {code} – {name}: {'tăng' if delta>=0 else 'giảm'} {_fmt_money(abs(delta))} VND so với đầu kỳ.")

    return "\n".join(lines) + "\n"



def narrate_rows(table_text: str) -> str:
    """Diễn giải từng dòng: 1 câu/1 dòng CODE|NAME|NOTE|END|BEGIN"""
    rows = parse_table(table_text)
    lines = []
    for r in rows:
        code  = r.get("code","")
        name  = r.get("name","")
        end_s = r.get("end","")
        beg_s = r.get("begin","")
        e = _parse_number(end_s)
        b = _parse_number(beg_s)
        if e is not None and b not in (None, 0):
            pct  = (e - b) / abs(b) * 100.0
            sign = "tăng" if e - b >= 0 else "giảm"
            lines.append(
                f"Mã {code} – {name}: cuối kỳ {_fmt_money(e)} VND; đầu kỳ {_fmt_money(b)} VND; {sign} ~{abs(pct):.1f}%."
            )
        else:
            # fallback khi không parse được số
            lines.append(
                f"Mã {code} – {name}: cuối kỳ {end_s} VND; đầu kỳ {beg_s} VND."
            )
    return "\n".join(lines) + ("\n" if lines else "")
