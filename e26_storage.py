"""
e26_storage.py
Local folder-based storage for vendor submissions (no DB).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class SubmissionMeta:
    project_id: str
    vendor: str
    submitted_at: str  # ISO8601
    submission_id: str  # filename stem


def _now_iso_local() -> str:
    # Use local time; Streamlit runs on your PC so this is fine.
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sanitize_name(s: str) -> str:
    s = (s or "").strip()
    # keep alnum, dash, underscore only
    out = []
    for ch in s:
        if ch.isalnum() or ch in ("-", "_"):
            out.append(ch)
        elif ch in (" ", ".", "@"):
            out.append("_")
        # drop others
    return "".join(out)[:80] or "UNKNOWN"


def get_data_root() -> Path:
    # Default root: ./data (relative to app working dir)
    return Path(os.environ.get("E26_DATA_ROOT", "data")).resolve()


def project_dir(project_id: str) -> Path:
    pid = sanitize_name(project_id)
    return get_data_root() / "projects" / pid


def submissions_dir(project_id: str) -> Path:
    return project_dir(project_id) / "submissions"


def ensure_dirs(project_id: str) -> None:
    submissions_dir(project_id).mkdir(parents=True, exist_ok=True)


def list_projects() -> List[str]:
    root = get_data_root() / "projects"
    if not root.exists():
        return []
    return sorted([p.name for p in root.iterdir() if p.is_dir()])


def list_submission_json(project_id: str) -> List[Path]:
    d = submissions_dir(project_id)
    if not d.exists():
        return []
    return sorted(d.glob("*.json"), reverse=True)


def save_submission(
    project_id: str,
    vendor: str,
    df_rec190: pd.DataFrame,
    result_rows: List[Dict],
    source_filename: Optional[str] = None,
    source_bytes: Optional[bytes] = None,
) -> SubmissionMeta:
    """
    Save:
      - raw file (optional): .xlsx or .csv
      - normalized rec190 data: .csv
      - results json: .json
    """
    ensure_dirs(project_id)

    pid = sanitize_name(project_id)
    ven = sanitize_name(vendor)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_id = f"{pid}__{ven}__{ts}"

    subdir = submissions_dir(project_id)

    # 1) raw file if provided
    raw_saved = None
    if source_bytes is not None:
        ext = ""
        if source_filename:
            ext = Path(source_filename).suffix.lower()
        if ext not in (".xlsx", ".xls", ".csv"):
            ext = ".bin"
        raw_saved = subdir / f"{submission_id}{ext}"
        raw_saved.write_bytes(source_bytes)

    # 2) rec190 normalized csv
    rec_csv = subdir / f"{submission_id}__rec190.csv"
    df_rec190.to_csv(rec_csv, index=False, encoding="utf-8-sig")

    # 3) result json
    payload = {
        "project_id": project_id,
        "vendor": vendor,
        "submitted_at": _now_iso_local(),
        "submission_id": submission_id,
        "source_filename": source_filename,
        "saved_raw": str(raw_saved) if raw_saved else None,
        "saved_rec190_csv": str(rec_csv),
        "rows": result_rows,
    }
    out_json = subdir / f"{submission_id}.json"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return SubmissionMeta(
        project_id=project_id,
        vendor=vendor,
        submitted_at=payload["submitted_at"],
        submission_id=submission_id,
    )


def load_project_aggregate(project_id: str) -> pd.DataFrame:
    """
    Aggregate all submissions json for a project into one DataFrame.
    Each row is one equipment/system row from vendor submission.
    """
    json_files = list_submission_json(project_id)
    all_rows: List[Dict] = []
    for jf in json_files:
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue
        vendor = data.get("vendor", "")
        submitted_at = data.get("submitted_at", "")
        submission_id = data.get("submission_id", jf.stem)
        for r in data.get("rows", []):
            row = dict(r)
            row["_vendor"] = vendor
            row["_submitted_at"] = submitted_at
            row["_submission_id"] = submission_id
            all_rows.append(row)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    # Put meta columns first if present
    preferred = [
        "_vendor", "_submitted_at", "_submission_id",
        "Row", "Item Number", "System", "Equipment",
        "System category", "Security zone",
        "E26 In Scope", "Scope Reason",
    ]
    meta_cols = [c for c in preferred if c in df.columns]
    other_cols = [c for c in df.columns if c not in meta_cols]
    return df[meta_cols + other_cols]


def export_aggregate_excel(df: pd.DataFrame) -> bytes:
    """
    Return Excel file bytes for download.
    """
    import io
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Aggregate")
    return bio.getvalue()
