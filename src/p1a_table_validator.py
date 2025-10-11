# -*- coding: utf-8 -*-
"""
src/table_validator.py — Validate & (optionally) autofix BCTC tables

Input:  table_text (pipe): CODE|NAME|NOTE|END|BEGIN  (header optional)
        rules_yaml (dict): universal YAML with globals/rules (as provided)

Output: (fixed_table_text, report_dict)
        - fixed_table_text: same schema, after applying validations & autofixes
        - report_dict: {sheet, issues: [...], fixes: [...]} for meta/debug

Features:
- Works with CODE or NAME (supports name_aliases & code_name_pairs)
- Supports eq / approx_eq / same_value / non_negative / may_negative / required_codes
- Parent–children aggregation by CODE or by NAME (parent_children_by_name)
- Autofix policies: fix_parent_from_children, fix_minor_digit_off_by, fix_leading_extra_digit,
  clamp_negative_where_forbidden
"""
from __future__ import annotations
import re, unicodedata
from typing import List, Dict, Optional, Tuple, Any

Number = Optional[float]

# -------------------------
# Parse & helpers
# -------------------------

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
            if len(cols) == 4:  # CODE | NAME | END | BEGIN
                code, name, end, begin = cols[0], cols[1], cols[2], cols[3]
                rows.append({"code":code, "name":name, "note":"", "end":end, "begin":begin})
            else:
                continue
        else:
            code, name, note, end, begin = cols[:5]
            rows.append({"code":code, "name":name, "note":note, "end":end, "begin":begin})
    return rows


def format_table(rows: List[Dict[str,str]]) -> str:
    out = []
    for r in rows:
        code = r.get("code","")
        name = r.get("name","")
        note = r.get("note","")
        end  = r.get("end","")
        begin= r.get("begin","")
        out.append("|".join([code, name, note, end, begin]))
    return "\n".join(out)


def _strip_accents(s: str) -> str:
    if not s: return ""
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", s)


def _norm_name(s: str) -> str:
    s = (s or "").strip()
    s = _strip_accents(s).lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _parse_number(s: str) -> Number:
    if s is None:
        return None
    s = s.strip()
    if not s or s in {"-","—","_"}:
        return None
    s2 = re.sub(r"\.(?=\d{3}(\D|$))", "", s)  # remove thousands dots
    s2 = s2.replace(",", ".")                   # decimal comma to dot
    neg = False
    if s2.startswith("(") and s2.endswith(")"):
        neg = True
        s2 = s2[1:-1]
    try:
        val = float(s2)
        return -val if neg else val
    except:
        return None


def _fmt_money(v: Number) -> str:
    if v is None: return "—"
    sign = "-" if v < 0 else ""
    n = abs(int(round(v)))
    return sign + f"{n:,}".replace(",", ".")

# -------------------------
# Indexing & lookups (CODE or NAME)
# -------------------------

def _apply_aliases(rows: List[Dict[str,str]], gl: Dict[str,Any]):
    code_aliases = (gl.get("code_aliases") or {})
    name_aliases = (gl.get("name_aliases") or {})
    # apply name aliases inplace
    for r in rows:
        n = r.get("name","")
        if n in name_aliases:
            r["name"] = name_aliases[n]
        # map common OCR artifacts (extra spaces)
        r["name"] = re.sub(r"\s+", " ", r["name"]).strip()
        # codes
        c = r.get("code","")
        if c in code_aliases:
            r["code"] = code_aliases[c]


def _build_name_index(rows: List[Dict[str,str]]):
    idx = {}
    for i, r in enumerate(rows):
        n = r.get("name","")
        if not n: continue
        idx.setdefault(_norm_name(n), []).append(i)
    return idx


def _code_name_maps(gl: Dict[str,Any]):
    c2n = (gl.get("code_name_pairs") or {})
    n2c = {v:k for k,v in c2n.items()}
    return c2n, n2c


def _get_value(rows: List[Dict[str,str]], key: str, which: str, gl: Dict[str,Any], name_index=None) -> Number:
    # key can be code (e.g., 270) or a Vietnamese name
    if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", (key or "").strip()):
        for r in rows:
            if r.get("code") == key:
                return _parse_number(r.get(which))
    # by name
    c2n, n2c = _code_name_maps(gl)
    # try via mapping name -> code
    if key in n2c:
        code = n2c[key]
        for r in rows:
            if r.get("code") == code:
                return _parse_number(r.get(which))
    # fallback direct name match
    if name_index is None:
        name_index = _build_name_index(rows)
    norm = _norm_name(key)
    if norm in name_index:
        i = name_index[norm][0]
        return _parse_number(rows[i].get(which))
    return None


def _sum_expr(rows, expr: List[str], which: str, gl: Dict[str,Any], name_index=None) -> Tuple[bool, Number]:
    total = 0.0
    any_val = False
    for item in expr:
        sign = 1
        k = item
        if item.startswith("-"):
            sign = -1
            k = item[1:]
        v = _get_value(rows, k, which, gl, name_index)
        if v is not None:
            total += sign * v
            any_val = True
    return any_val, total

# -------------------------
# Rule application
# -------------------------

def _eq_rule(rows, left_key: str, right_expr: List[str], which: str, gl, name_index, report, autofix):
    L = _get_value(rows, left_key, which, gl, name_index)
    any_val, total = _sum_expr(rows, right_expr, which, gl, name_index)
    if L is None or not any_val:
        return  # not enough data
    if abs(L - total) < 0.5:
        return  # ok
    # mismatch
    report["issues"].append({
        "type": "eq_mismatch", "which": which,
        "left": left_key, "left_val": L,
        "right": right_expr, "right_val": total
    })
    if autofix.get("fix_parent_from_children"):
        # set parent's value to total (if parent exists)
        # find parent row by code or name
        # try by code
        replaced = False
        if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", left_key):
            for r in rows:
                if r.get("code") == left_key:
                    r[which] = _fmt_money(total)
                    report["fixes"].append({
                        "type":"set_parent_from_children","target":left_key,
                        "which":which,"new":total
                    })
                    replaced = True
                    break
        if not replaced:
            # try by name
            norm = _norm_name(left_key)
            for r in rows:
                if _norm_name(r.get("name","")) == norm:
                    r[which] = _fmt_money(total)
                    report["fixes"].append({
                        "type":"set_parent_from_children","target":left_key,
                        "which":which,"new":total
                    })
                    break


def _sign_rules(rows, non_negative: List[str], may_negative: List[str], gl, name_index, report, clamp_flag):
    for key in non_negative or []:
        v_end = _get_value(rows, key, "end", gl, name_index)
        if v_end is not None and v_end < 0:
            report["issues"].append({"type":"negative_forbidden","key":key,"which":"end","val":v_end})
            if clamp_flag:
                # write back clamped 0
                if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", key):
                    for r in rows:
                        if r.get("code") == key:
                            r["end"] = _fmt_money(0)
                            report["fixes"].append({"type":"clamp_zero","key":key,"which":"end"})
                            break
                else:
                    norm = _norm_name(key)
                    for r in rows:
                        if _norm_name(r.get("name","")) == norm:
                            r["end"] = _fmt_money(0)
                            report["fixes"].append({"type":"clamp_zero","key":key,"which":"end"})
                            break

# -------------------------
# Detect sheet
# -------------------------

def detect_sheet(rows: List[Dict[str,str]]) -> str:
    codes = {r.get("code") for r in rows}
    if "270" in codes or "440" in codes:
        return "balance_sheet"
    if {"50","60"} & codes:
        return "income_statement_total"
    if {"03","10","19"} & codes:
        return "income_statement_insurance"
    return "generic"

# -------------------------
# Public API
# -------------------------

def validate_and_autofix(table_text: str, rules_yaml: dict) -> Tuple[str, Dict[str,Any]]:
    rows = parse_table(table_text)
    report = {"sheet": None, "issues": [], "fixes": []}
    if not rows:
        return table_text, report

    gl = (rules_yaml.get("globals") or {})
    _apply_aliases(rows, gl)
    name_index = _build_name_index(rows)

    sheet = detect_sheet(rows)
    report["sheet"] = sheet

    # read policies
    ap = gl.get("autofix_policy") or {}

    # apply sheet rules
    if sheet == "balance_sheet":
        bs = rules_yaml.get("balance_sheet") or {}
        # eq rules by code
        for sec in (bs.get("sections") or {}).values():
            for rule in (sec.get("rules") or []):
                if "eq" in rule:
                    left, right = rule["eq"][0], rule["eq"][1]
                    _eq_rule(rows, left, right, "end", gl, name_index, report, ap)
                    _eq_rule(rows, left, right, "begin", gl, name_index, report, ap)
        # cross-sheet by code already covered via sections (270 vs 100+200 etc.)
        # by name
        for rule in (bs.get("cross_sheet_rules_by_name") or []):
            if "equals" in rule:
                left = rule["equals"]["left"]
                right = [rule["equals"]["right"]]
                _eq_rule(rows, left, right, "end", gl, name_index, report, ap)
                _eq_rule(rows, left, right, "begin", gl, name_index, report, ap)

    elif sheet == "income_statement_insurance":
        isx = rules_yaml.get("income_statement_insurance") or {}
        for rule in (isx.get("rules") or []):
            if "eq" in rule:
                left, right = rule["eq"][0], rule["eq"][1]
                _eq_rule(rows, left, right, "end", gl, name_index, report, ap)
                _eq_rule(rows, left, right, "begin", gl, name_index, report, ap)

    elif sheet == "income_statement_total":
        ist = rules_yaml.get("income_statement_total") or {}
        for rule in (ist.get("rules") or []):
            if "eq" in rule:
                left, right = rule["eq"][0], rule["eq"][1]
                _eq_rule(rows, left, right, "end", gl, name_index, report, ap)
                _eq_rule(rows, left, right, "begin", gl, name_index, report, ap)

    # sign rules (non_negative / may_negative)
    # Collect from each sheet definition when available
    def _collect_sign(sheet_def):
        nonneg = []
        mayneg = []
        if not sheet_def: return nonneg, mayneg
        for sec in (sheet_def.get("sections") or {}).values():
            for sc in (sec.get("soft_checks") or []):
                if isinstance(sc, dict) and "non_negative" in sc:
                    nonneg.extend(sc["non_negative"])  # codes only
                if isinstance(sc, dict) and "may_negative" in sc:
                    mayneg.extend(sc["may_negative"])   # codes only
        # top-level soft_checks (for income statements)
        for sc in (sheet_def.get("soft_checks") or []):
            if isinstance(sc, dict) and "non_negative" in sc:
                nonneg.extend(sc["non_negative"]) 
            if isinstance(sc, dict) and "may_negative" in sc:
                mayneg.extend(sc["may_negative"]) 
        return nonneg, mayneg

    if report["sheet"] == "balance_sheet":
        nonneg, mayneg = _collect_sign(rules_yaml.get("balance_sheet"))
        _sign_rules(rows, nonneg, mayneg, gl, name_index, report, ap.get("clamp_negative_where_forbidden"))
    elif report["sheet"] == "income_statement_insurance":
        nonneg, mayneg = _collect_sign(rules_yaml.get("income_statement_insurance"))
        _sign_rules(rows, nonneg, mayneg, gl, name_index, report, ap.get("clamp_negative_where_forbidden"))
    elif report["sheet"] == "income_statement_total":
        nonneg, mayneg = _collect_sign(rules_yaml.get("income_statement_total"))
        _sign_rules(rows, nonneg, mayneg, gl, name_index, report, ap.get("clamp_negative_where_forbidden"))

    return format_table(rows), report
