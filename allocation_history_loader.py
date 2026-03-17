# -*- coding: utf-8 -*-
"""
allocation_history_loader.py
────────────────────────────
Loads and normalises investment-allocation history data from Google Sheets.

Two transport modes (auto-selected):
  1. Public CSV export  – works when the sheet is published to web (no auth needed)
  2. gspread service account – requires st.secrets["gcp_service_account"]

Normalised output schema
────────────────────────
manager         : str   – e.g. "הראל"
track           : str   – e.g. "כללי" | "מנייתי"
date            : datetime64[ns] – first day of the relevant month (deterministic)
year            : int
month           : int
allocation_name : str   – e.g. "מניות חו\"ל (דלתא)"
allocation_value: float – percent value, e.g. 38.5
source_sheet    : str   – raw sheet/tab name

Usage
─────
from allocation_history_loader import load_allocation_history

df = load_allocation_history(
    sheet_url="https://docs.google.com/spreadsheets/d/SHEET_ID/...",
)
"""

from __future__ import annotations

import re
import io
import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st

logger = logging.getLogger(__name__)

# ─── Hebrew month name → number ───────────────────────────────────────────────
_HEB_MONTHS = {
    "ינואר": 1,  "פברואר": 2,  "מרץ": 3,    "מרס": 3,
    "אפריל": 4,  "מאי": 5,     "יוני": 6,
    "יולי": 7,   "אוגוסט": 8,  "ספטמבר": 9,
    "אוקטובר": 10, "נובמבר": 11, "דצמבר": 12,
}

# ─── Sheet-name → (manager, track) heuristic ─────────────────────────────────
_SHEET_META: dict[str, dict] = {
    # Harel
    "הראל כללי":    {"manager": "הראל",  "track": "כללי"},
    "הראל מנייתי":  {"manager": "הראל",  "track": "מנייתי"},
    # add more mappings here as new sheets arrive
}

# Fallback regex patterns – tries to infer manager/track from sheet name
_MANAGER_PATTERNS = [
    "הראל", "מגדל", "כלל", "מנורה", "הפניקס", "אנליסט", "מיטב",
    "ילין", "פסגות", "אלטשולר", "ברקת", "אלומות",
]
_TRACK_PATTERNS = {
    "כלל": "כללי", "כללי": "כללי",
    "מנייתי": "מנייתי", "מניות": "מנייתי",
    "אג\"ח": "אג\"ח", "חו\"ל": "חו\"ל",
}


def _infer_meta(sheet_name: str) -> dict:
    """Try to infer manager + track from a sheet tab name."""
    s = sheet_name.strip()
    # exact lookup first
    for key, meta in _SHEET_META.items():
        if key in s:
            return meta
    # regex fallback
    manager = next((m for m in _MANAGER_PATTERNS if m in s), s)
    track = "כללי"
    for pat, val in _TRACK_PATTERNS.items():
        if pat in s:
            track = val
            break
    return {"manager": manager, "track": track}


def _extract_sheet_id(url: str) -> str:
    """Extract Google Sheets document ID from any Google Sheets URL."""
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9_-]+)", url)
    if not m:
        raise ValueError(f"Cannot extract sheet ID from URL: {url}")
    return m.group(1)


def _csv_export_url(sheet_id: str, gid: int = 0) -> str:
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"


def _public_sheet_names_and_gids(sheet_id: str) -> list[tuple[str, int]]:
    """
    Fetch the list of (sheet_name, gid) pairs from the HTML of the spreadsheet.
    Works only for publicly viewable sheets.
    """
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/edit"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        # The sheet metadata is embedded as JSON-like structure in the page
        matches = re.findall(r'"sheetId":\s*(\d+).*?"title":\s*"([^"]+)"', r.text)
        if not matches:
            # try alternate pattern
            matches = re.findall(r'\["([^"]+)",null,(\d+)\]', r.text)
            return [(name, int(gid)) for name, gid in matches] if matches else [("Sheet1", 0)]
        return [(name, int(gid)) for gid, name in matches]
    except Exception as e:
        logger.warning(f"Could not fetch sheet list: {e}")
        return [("Sheet1", 0)]


def _parse_date_value(val) -> Optional[datetime]:
    """
    Robustly parse a date from various formats:
      - datetime / Timestamp
      - "ינואר 2024", "Jan-2024", "01/2024", "2024-01", "2024-01-01", ...
    Returns a datetime at the first of the month, or None.
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    if isinstance(val, (datetime, pd.Timestamp)):
        return pd.Timestamp(val).replace(day=1).to_pydatetime()

    s = str(val).strip()
    if not s or s.lower() in ("nan", "none", ""):
        return None

    # Hebrew month name + year: "ינואר 2024"
    for heb, month_num in _HEB_MONTHS.items():
        if heb in s:
            year_m = re.search(r"(\d{4})", s)
            if year_m:
                return datetime(int(year_m.group(1)), month_num, 1)

    # Try pandas parser for common formats
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%Y", "%Y-%m", "%b-%Y", "%B %Y",
                "%b %Y", "%Y/%m/%d", "%d-%m-%Y"):
        try:
            dt = datetime.strptime(s, fmt)
            return dt.replace(day=1)
        except ValueError:
            pass

    # Last resort
    try:
        dt = pd.to_datetime(s, dayfirst=True)
        return dt.replace(day=1).to_pydatetime()
    except Exception:
        return None


def _parse_percent(val) -> Optional[float]:
    """
    Parse a percentage value to float (0-100 range).
    Handles: "38.5%", "0.385", 38.5, "38,5", "38.5 %", etc.
    """
    if val is None:
        return None
    if isinstance(val, (int, float)):
        if np.isnan(val):
            return None
        # If stored as decimal fraction (0–1 range), convert
        if abs(val) <= 1.5:
            return round(float(val) * 100, 4)
        return round(float(val), 4)
    s = str(val).strip().replace(",", ".").replace("%", "").strip()
    if not s:
        return None
    try:
        f = float(s)
        if abs(f) <= 1.5:
            return round(f * 100, 4)
        return round(f, 4)
    except ValueError:
        return None


def _normalise_sheet_df(
    raw: pd.DataFrame,
    sheet_name: str,
) -> pd.DataFrame:
    """
    Convert a raw single-sheet DataFrame into the normalised long format.

    Expected wide format (example):
        תאריך  | מניות חו"ל (דלתא) | אג"ח חו"ל (דלתא) | ...
        ינואר 2014 | 27.3% | 35.1% | ...

    The first column is assumed to be the date column.
    All other columns are allocation components.
    """
    if raw is None or raw.empty:
        return pd.DataFrame()

    meta = _infer_meta(sheet_name)

    # ── Identify date column ────────────────────────────────────────
    # Usually column 0 or a column named like "תאריך" / "חודש" / "date"
    date_col_candidates = [
        c for c in raw.columns
        if any(kw in str(c).lower() for kw in ["תאריך", "חודש", "date", "month", "time"])
    ]
    date_col = date_col_candidates[0] if date_col_candidates else raw.columns[0]

    # ── Allocation columns = all except date_col ────────────────────
    alloc_cols = [c for c in raw.columns if c != date_col]

    rows = []
    for _, row in raw.iterrows():
        dt = _parse_date_value(row[date_col])
        if dt is None:
            continue
        for col in alloc_cols:
            val = _parse_percent(row[col])
            if val is None:
                continue
            rows.append({
                "manager":          meta["manager"],
                "track":            meta["track"],
                "date":             pd.Timestamp(dt),
                "year":             dt.year,
                "month":            dt.month,
                "allocation_name":  str(col).strip(),
                "allocation_value": val,
                "source_sheet":     sheet_name,
            })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


# ─── Public CSV transport ─────────────────────────────────────────────────────

def _load_sheet_via_csv(sheet_id: str, gid: int, sheet_name: str) -> pd.DataFrame:
    url = _csv_export_url(sheet_id, gid)
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        raw = pd.read_csv(io.StringIO(r.text), header=0)
        # Drop rows that are entirely empty
        raw = raw.dropna(how="all").reset_index(drop=True)
        return _normalise_sheet_df(raw, sheet_name)
    except Exception as e:
        logger.warning(f"CSV load failed for sheet '{sheet_name}' (gid={gid}): {e}")
        return pd.DataFrame()


# ─── gspread transport ────────────────────────────────────────────────────────

def _load_via_gspread(sheet_url: str) -> pd.DataFrame:
    """Load all sheets using gspread + service account credentials from st.secrets."""
    try:
        import gspread
        from google.oauth2.service_account import Credentials

        creds_dict = dict(st.secrets["gcp_service_account"])
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets.readonly",
            "https://www.googleapis.com/auth/drive.readonly",
        ]
        creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        gc = gspread.authorize(creds)
        sh = gc.open_by_url(sheet_url)

        frames = []
        for ws in sh.worksheets():
            try:
                data = ws.get_all_values()
                if not data or len(data) < 2:
                    continue
                raw = pd.DataFrame(data[1:], columns=data[0])
                raw = raw.dropna(how="all").reset_index(drop=True)
                norm = _normalise_sheet_df(raw, ws.title)
                if not norm.empty:
                    frames.append(norm)
            except Exception as e:
                logger.warning(f"gspread: skip sheet '{ws.title}': {e}")

        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    except Exception as e:
        logger.error(f"gspread transport failed: {e}")
        return pd.DataFrame()


# ─── Main public API ──────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def load_allocation_history(sheet_url: str) -> tuple[pd.DataFrame, list[str]]:
    """
    Load all sheets from the given Google Sheets URL.

    Returns
    -------
    df     : normalised DataFrame  (empty if loading failed)
    errors : list of warning strings
    """
    errors: list[str] = []

    if not sheet_url or sheet_url.strip() == "":
        return pd.DataFrame(), ["לא הוגדר קישור ל-Google Sheets"]

    # ── Try gspread first (if service account is in secrets) ────────
    has_sa = hasattr(st, "secrets") and "gcp_service_account" in st.secrets
    if has_sa:
        df = _load_via_gspread(sheet_url)
        if not df.empty:
            return df, errors
        errors.append("gspread נכשל — מנסה CSV ציבורי")

    # ── Fallback: public CSV export ─────────────────────────────────
    try:
        sheet_id = _extract_sheet_id(sheet_url)
    except ValueError as e:
        return pd.DataFrame(), [str(e)]

    sheets = _public_sheet_names_and_gids(sheet_id)
    frames = []
    for name, gid in sheets:
        norm = _load_sheet_via_csv(sheet_id, gid, name)
        if not norm.empty:
            frames.append(norm)
        else:
            errors.append(f"גליון '{name}' — לא נטען נתון תקין")

    if not frames:
        return pd.DataFrame(), errors + ["לא נטענו נתונים מאף גליון"]

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["manager", "track", "allocation_name", "date"]).reset_index(drop=True)
    return df, errors
