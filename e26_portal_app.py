"""
e26_portal_app.py
Streamlit single-file app with two tabs:
- Vendor: upload Rec.190 (xlsx/csv), run IACS scope check, submit (save to local folder)
- Admin: choose project (Hull No.), aggregate all submissions into one table + export

Run:
  streamlit run e26_portal_app.py

Local storage root:
  ./data/projects/<PROJECT_ID>/submissions/

Optional:
  set env var E26_DATA_ROOT to change storage root.
"""

from __future__ import annotations

import io
from typing import Dict, List

import pandas as pd
import streamlit as st

from e26_storage import (
    export_aggregate_excel,
    get_data_root,
    list_projects,
    load_project_aggregate,
    save_submission,
)

# Import classifier if available; if not, app still runs but without scope judgement.
SCOPE_AVAILABLE = True
try:
    from e26_classifier import E26Classifier
except Exception:
    SCOPE_AVAILABLE = False


# ===== Rec.190 columns (A~V) =====
REC190_COLUMNS = [
    "Item Number",
    "Ship functions and systems",
    "System",
    "Equipment",
    "System category",
    "Brand/Manufacturer",
    "Model/Type",
    "Unique Identifier",
    "Operating System (OS)",
    "Firmware",
    "Software and version",
    "Location Onboard",
    "Security zone",
    "Ship description of functionality/purpose",
    "Connection to components and systems with the scope",
    "Connections to Untrusted networks",
    "Physical interfaces",
    "Supported communication protocols",
    "IP ranges/MAC address of nodes connected (if applicable)",
    "Certificate : Type Approval, etc., (if applicable)",
    "Excluded (if part of an excluded system in accordance with UR E26, Sec 6)",
    "Update log",
]


def make_rec190_template_df(rows: int = 50) -> pd.DataFrame:
    data = [{c: "" for c in REC190_COLUMNS} for _ in range(rows)]
    return pd.DataFrame(data, columns=REC190_COLUMNS)


def read_rec190_file(upload) -> pd.DataFrame:
    """
    Read xlsx/csv into DataFrame, normalize to REC190_COLUMNS (missing cols will be created).
    """
    name = (upload.name or "").lower()
    raw = upload.getvalue()
    if name.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(raw), dtype=str, keep_default_na=False)
    else:
        # xlsx/xls
        df = pd.read_excel(io.BytesIO(raw), dtype=str, keep_default_na=False)

    # Normalize columns: trim spaces
    df.columns = [str(c).strip() for c in df.columns]

    # Ensure required columns exist
    for c in REC190_COLUMNS:
        if c not in df.columns:
            df[c] = ""

    # Keep only known columns (and preserve order)
    df = df[REC190_COLUMNS].copy()

    # Replace NaN -> ""
    df = df.fillna("")
    return df


def run_scope_checks(df: pd.DataFrame) -> List[Dict]:
    """
    Return list of dict rows including scope judgement.
    Each dict is an output record saved to JSON.
    """
    results: List[Dict] = []

    clf = None
    if SCOPE_AVAILABLE:
        try:
            clf = E26Classifier()
        except Exception:
            clf = None

    records = df.to_dict(orient="records")

    for idx, row in enumerate(records, start=1):
        # Skip empty lines
        if not str(row.get("System", "")).strip() and not str(row.get("Equipment", "")).strip():
            continue

        decision_in_scope = None
        reasons: List[str] = []
        if clf is not None and hasattr(clf, "is_e26_in_scope_rec190"):
            try:
                decision_in_scope, reasons = clf.is_e26_in_scope_rec190(row)
            except Exception as e:
                decision_in_scope = None
                reasons = [f"Scope check error: {e}"]
        else:
            decision_in_scope = None
            reasons = ["Scope check not available (classifier missing or function not found)"]

        results.append({
            "Row": idx,
            "Item Number": row.get("Item Number", ""),
            "System": row.get("System", ""),
            "Equipment": row.get("Equipment", ""),
            "System category": row.get("System category", ""),
            "Security zone": row.get("Security zone", ""),
            "Connections to Untrusted networks": row.get("Connections to Untrusted networks", ""),
            "E26 In Scope": ("대상" if decision_in_scope else "비대상") if decision_in_scope is not None else "판정불가",
            "Scope Reason": " | ".join(reasons),
            "rec190": row,  # keep full original row for admin deep-dive
        })

    return results


def vendor_tab():
    st.header("Vendor 제출 페이지")
    st.caption("프로젝트(호선) 단위로 제출하면, 씨넷은 관리자 페이지에서 호선별로 자동 집계합니다.")

    # Inputs
    col1, col2 = st.columns([1, 1])
    with col1:
        project_id = st.text_input("프로젝트 ID (호선)", placeholder="예: H3456 / SN2345")
    with col2:
        vendor = st.text_input("Vendor (회사명)", placeholder="예: ABC Marine Systems")

    st.divider()

    st.subheader("Rec.190 입력 방식")
    mode = st.radio("입력 방식 선택", ["엑셀/CSV 업로드", "웹 표 입력(50행 템플릿)"], horizontal=True)

    df: pd.DataFrame | None = None
    source_name = None
    source_bytes = None

    if mode == "엑셀/CSV 업로드":
        upload = st.file_uploader("Rec.190 형식 파일 업로드 (.xlsx 또는 .csv)", type=["xlsx", "xls", "csv"])
        if upload is not None:
            try:
                df = read_rec190_file(upload)
                source_name = upload.name
                source_bytes = upload.getvalue()
                st.success(f"파일 로드 완료: {upload.name} (rows: {len(df)})")
            except Exception as e:
                st.error(f"파일을 읽을 수 없습니다: {e}")
                df = None
    else:
        # Form to avoid per-keystroke reruns causing “double typing”
        if "rec190_web_df" not in st.session_state:
            st.session_state["rec190_web_df"] = make_rec190_template_df(rows=50)

        with st.form("rec190_web_form", clear_on_submit=False):
            st.info("표 편집 후 **입력 반영**을 눌러야 저장됩니다. (입력 튕김 방지)")
            edited = st.data_editor(
                st.session_state["rec190_web_df"],
                use_container_width=True,
                num_rows="dynamic",
                height=650,
                key="rec190_web_editor",
            )
            c1, c2, _ = st.columns([1, 1, 2])
            with c1:
                apply_btn = st.form_submit_button("입력 반영")
            with c2:
                reset_btn = st.form_submit_button("표 초기화(50행)")
        if apply_btn:
            st.session_state["rec190_web_df"] = edited
            st.success("입력이 반영되었습니다.")
        if reset_btn:
            st.session_state["rec190_web_df"] = make_rec190_template_df(rows=50)
            st.success("표를 초기화했습니다.")
            st.rerun()

        df = st.session_state["rec190_web_df"].copy()

    st.divider()

    if df is None:
        st.warning("입력(업로드/표)을 먼저 준비해 주세요.")
        return

    st.subheader("제출 전 미리보기")
    st.dataframe(df.head(20), use_container_width=True)

    run_btn = st.button("E26 판정 실행(미리보기)", key="vendor_run_preview")
    if run_btn:
        results = run_scope_checks(df)
        if not results:
            st.warning("유효한 행이 없습니다. 최소 System 또는 Equipment 입력 필요.")
            return
        res_df = pd.DataFrame([{
            "Row": r["Row"],
            "Item Number": r["Item Number"],
            "System": r["System"],
            "Equipment": r["Equipment"],
            "E26 In Scope": r["E26 In Scope"],
            "Scope Reason": r["Scope Reason"],
        } for r in results])

        st.success(f"판정 완료: {len(res_df)} rows")
        st.dataframe(res_df, use_container_width=True)

        st.session_state["last_results"] = results
        st.session_state["last_project_id"] = project_id
        st.session_state["last_vendor"] = vendor
        st.session_state["last_df"] = df

    st.divider()

    st.subheader("씨넷에 제출(업로드)")
    st.caption("제출 시 로컬 폴더에 저장됩니다. (DB 없음)")

    can_submit = (
        bool(project_id.strip()) and
        bool(vendor.strip()) and
        ("last_results" in st.session_state) and
        (st.session_state.get("last_project_id") == project_id) and
        (st.session_state.get("last_vendor") == vendor)
    )

    if not can_submit:
        st.warning("제출하려면: (1) 프로젝트ID/벤더 입력 (2) E26 판정 실행(미리보기) 를 먼저 해주세요.")
        return

    submit_btn = st.button("제출 저장", key="vendor_submit")
    if submit_btn:
        meta = save_submission(
            project_id=project_id,
            vendor=vendor,
            df_rec190=st.session_state["last_df"],
            result_rows=st.session_state["last_results"],
            source_filename=source_name,
            source_bytes=source_bytes,
        )
        st.success(f"저장 완료: {meta.submission_id}")

        st.info(f"저장 루트: {get_data_root()}")
        st.info("Admin 탭에서 프로젝트를 선택하면, 방금 제출한 결과가 자동 집계됩니다.")


def admin_tab():
    st.header("Admin (씨넷 관리자 페이지)")
    st.caption("호선(Project ID)별로 벤더 제출물을 자동 집계하여 한 표로 확인/정렬/내보내기 합니다.")

    projects = list_projects()
    if not projects:
        st.warning("저장된 프로젝트가 없습니다. Vendor 탭에서 먼저 제출을 저장해 주세요.")
        st.info(f"저장 루트: {get_data_root()}")
        return

    project_id = st.selectbox("프로젝트(호선) 선택", projects, index=0)

    df = load_project_aggregate(project_id)
    if df.empty:
        st.warning("해당 프로젝트에 제출물이 없습니다.")
        return

    st.subheader("필터")
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        scope_opts = sorted(df["E26 In Scope"].dropna().unique().tolist())
        scope_filter = st.multiselect("E26 In Scope", options=scope_opts, default=None)
    with c2:
        vendor_opts = sorted(df["_vendor"].dropna().unique().tolist())
        vendor_filter = st.multiselect("Vendor", options=vendor_opts, default=None)
    with c3:
        keyword = st.text_input("키워드 검색 (System/Equipment/Reason)", placeholder="예: ICMS, firewall, Windows")

    filtered = df.copy()
    if scope_filter:
        filtered = filtered[filtered["E26 In Scope"].isin(scope_filter)]
    if vendor_filter:
        filtered = filtered[filtered["_vendor"].isin(vendor_filter)]
    if keyword.strip():
        k = keyword.strip().lower()
        cols = [c for c in ["System", "Equipment", "Scope Reason"] if c in filtered.columns]
        mask = None
        for c in cols:
            m = filtered[c].astype(str).str.lower().str.contains(k, na=False)
            mask = m if mask is None else (mask | m)
        if mask is not None:
            filtered = filtered[mask]

    st.subheader(f"프로젝트 집계 표: {project_id}")
    st.dataframe(filtered, use_container_width=True, height=650)

    st.subheader("Export")
    xls_bytes = export_aggregate_excel(filtered)
    st.download_button(
        "프로젝트 집계 엑셀 다운로드",
        data=xls_bytes,
        file_name=f"{project_id}_aggregate.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="dl_aggregate",
    )

    st.caption(f"저장 위치: {get_data_root() / 'projects' / project_id / 'submissions'}")


def main():
    st.set_page_config(page_title="E26 Vendor/Admin Portal (No-DB)", layout="wide")
    st.title("E26 Vendor/Admin Portal (No DB, Local Folder Storage)")

    if not SCOPE_AVAILABLE:
        st.warning("주의: e26_classifier import 실패. 판정 기능 없이 저장/집계만 동작합니다.")

    tab_vendor, tab_admin = st.tabs(["Vendor 제출", "Admin 집계"])

    with tab_vendor:
        vendor_tab()

    with tab_admin:
        admin_tab()


if __name__ == "__main__":
    main()
