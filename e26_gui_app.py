import re
import streamlit as st

from e26_classifier import EquipmentInput, E26Classifier
import pandas as pd

import e26_classifier
st.write("DEBUG e26_classifier path:", e26_classifier.__file__)


# =========================
# Rec.190 Vessel Asset Inventory Columns (A~V)
# =========================
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
    """
    Rec.190 Vessel Asset Inventory 입력용 템플릿 (기본 50행)
    """
    data = [{c: "" for c in REC190_COLUMNS} for _ in range(rows)]
    return pd.DataFrame(data)

def _to_bool_untrusted(v) -> bool:
    """
    Rec.190 'Connections to Untrusted networks' 값이 비어있지 않으면 True로 간주(보수적)
    - 'Y/Yes/True/1' 명시도 지원
    - 비어있으면 False
    """
    if v is None:
        return False
    s = str(v).strip().lower()
    if not s:
        return False
    if s in ("n", "no", "false", "0"):
        return False
    return True


def build_equipment_input_from_rec190_row(row: dict, row_no: int) -> EquipmentInput:
    """
    Rec.190 표의 1행(dict)을 EquipmentInput으로 변환
    - name: Equipment 우선, 없으면 System
    - vendor: Brand/Manufacturer
    - function_hint: Ship functions and systems
    - description: 모든 필드를 'key: value'로 합쳐 분류 힌트로 제공
    """
    system_name = str(row.get("System", "")).strip()
    equip_name = str(row.get("Equipment", "")).strip()
    vendor = str(row.get("Brand/Manufacturer", "")).strip()

    name = equip_name or system_name or f"(Row {row_no})"

    desc_parts = []
    for k in REC190_COLUMNS:
        v = row.get(k, "")
        if v is None:
            continue
        s = str(v).strip()
        if not s:
            continue
        desc_parts.append(f"{k}: {s}")
    description = " | ".join(desc_parts)

    eq = EquipmentInput(
        id=str(row.get("Item Number") or row_no),
        name=name,
        vendor=vendor,
        system=system_name,
        function_hint=str(row.get("Ship functions and systems", "")).strip() or None,
        description=description,
        is_computer_based=None,
        controls_physical_process=None,
        network_zone=str(row.get("Security zone", "")).strip() or None,
        connected_to_external_network=_to_bool_untrusted(row.get("Connections to Untrusted networks", "")),
    )
    return eq

def parse_lines(raw_text: str):
    """
    사용자 입력 텍스트를 줄 단위로 나누고,
    각 줄을 [system_name, full_line_text] 형태로 파싱한다.
    - system_name: 첫 컬럼(탭/콤마/세미콜론 기준)을 우선 시스템/장비명으로 사용
    - full_line_text: 전체 라인 (description으로 사용)
    """
    lines = raw_text.splitlines()
    items = []
    for idx, line in enumerate(lines, start=1):
        if not line.strip():
            continue

        # 탭 / 콤마 / 세미콜론 기준 분할
        parts = re.split(r"[\t,;]", line)
        if parts and parts[0].strip():
            system_name = parts[0].strip()
        else:
            system_name = line.strip()

        items.append({
            "id": idx,
            "system_name": system_name,
            "raw_line": line
        })
    return items


def build_equipment_input(item) -> EquipmentInput:
    """
    파싱된 한 줄 정보를 E26Classifier에서 쓰는 EquipmentInput 형태로 변환.
    여기서는 구조를 최대한 단순하게:
    - name/system: system_name
    - description: raw_line 전체
    - 나머지는 정보 부족 → None/False
    """
    system_name = item["system_name"]
    raw_line = item["raw_line"]

    eq = EquipmentInput(
        id=str(item["id"]),
        name=system_name,
        vendor="",                      # 텍스트에서 자동 추출은 추측이라 비워둠
        system=system_name,             # system 이름도 일단 동일하게
        function_hint=None,             # function은 텍스트 내용에서 classifier가 알아서 추정
        description=raw_line,           # 전체 라인을 description으로 사용
        is_computer_based=None,         # CBS 여부도 classifier 휴리스틱에 맡김
        controls_physical_process=None,
        network_zone=None,             # 위치 정보도 텍스트 내 키워드로 classifier가 어느정도 추론
        connected_to_external_network=False  # 외부망은 정보 부족 → 기본 False로 두고 보수적으로
    )
    return eq


def main():
    st.set_page_config(page_title="UR E26 대상 장비 판정 도우미", layout="wide")

    st.title("IACS Rec.190 기반 Vessel Asset Inventory")
    st.write(
        """
        - 아래 표(Rec.190 형식)에 장비 정보를 입력한 뒤 **[입력 반영]** 또는 바로 **[UR E26 판정 실행]**을 누르세요.  
        - 엑셀 그대로 복사/붙여넣기 가능합니다. 정보를 모두 입력한 뒤, [E26 판정 실행] 버튼을 누르면 대상장비인지 예상 정보가 출력됩니다.
        - 해당 페이지에서 표시되는 대상장비의 여부가 법적 근거나, 제출되는 답변이 될 수 없습니다. (Vendor 참고용입니다.) 
        """
    )

    DATA_KEY = "rec190_data"
    NONCE_KEY = "rec190_editor_nonce"

    # ✅ 타입 가드: 이전 실행에서 session_state가 list 등으로 오염된 경우 자동 복구
    if (DATA_KEY not in st.session_state) or (not isinstance(st.session_state[DATA_KEY], pd.DataFrame)):
        st.session_state[DATA_KEY] = make_rec190_template_df(rows=50)

    st.subheader("Vessel Asset Inventory (IACS Rec.190)")

    # nonce는 '초기화' 시에만 위젯을 새로 만들기 위한 장치
    nonce = st.session_state.get(NONCE_KEY, 0)

    with st.form("rec190_form", clear_on_submit=False):
        edited_df = st.data_editor(
            st.session_state[DATA_KEY],
            use_container_width=True,
            num_rows="dynamic",
            height=700,
            key=f"rec190_editor_widget_{nonce}",
        )

        c_apply, c_reset = st.columns([1, 1])
        with c_apply:
            apply_clicked = st.form_submit_button("입력 반영", use_container_width=True)
        with c_reset:
            reset_clicked = st.form_submit_button("표 초기화", use_container_width=True)

    if apply_clicked:
        st.session_state[DATA_KEY] = edited_df
        st.success("입력이 반영되었습니다.")

    if reset_clicked:
        st.session_state[DATA_KEY] = make_rec190_template_df(rows=50)
        st.session_state[NONCE_KEY] = st.session_state.get(NONCE_KEY, 0) + 1
        st.success("표가 초기화되었습니다.")
        st.rerun()

    # ✅ 판정 실행 버튼은 form 밖 (입력 확정/초기화와 분리)
    run_clicked = st.button("UR E26 판정 실행 (Rec.190 표 기반)", key="btn_run_rec190")

    if not run_clicked:
        return

    clf = E26Classifier()

    # 사용자가 '입력 반영'을 안 눌러도 현재 편집된 내용을 그대로 판정에 사용
    records = edited_df.to_dict(orient="records")

    # 빈 행 제거: System/Equipment 둘 다 비면 스킵
    valid_rows = []
    for idx, row in enumerate(records, start=1):
        if not str(row.get("System", "")).strip() and not str(row.get("Equipment", "")).strip():
            continue
        valid_rows.append((idx, row))

    if not valid_rows:
        st.warning("유효한 행이 없습니다. 최소 'System' 또는 'Equipment'를 입력해 주세요.")
        return

    results = []
    for row_no, row in valid_rows:
        # (1) 규정 기반 Scope 판정 (최종)
        decision_in_scope, reasons = clf.is_e26_in_scope_rec190(row)

        # (2) 기술 분류(참고용)
        eq = build_equipment_input_from_rec190_row(row, row_no)
        decision = clf.classify(eq)

        results.append({
            "Row": row_no,
            "Item Number": row.get("Item Number", ""),
            "Ship functions and systems": row.get("Ship functions and systems", ""),
            "System": row.get("System", ""),
            "Equipment": row.get("Equipment", ""),
            "System category": row.get("System category", ""),
            "Brand/Manufacturer": row.get("Brand/Manufacturer", ""),
            "Model/Type": row.get("Model/Type", ""),

            "E26 In Scope": "대상" if decision_in_scope else "비대상",
            "Scope Reason": " | ".join(reasons),

            "CBS": "CBS" if decision.is_cbs else "비CBS",
            "Function Class": decision.function_class or "",
            "Score": f"{decision.score:.2f}",
            "Confidence": f"{decision.confidence:.2f}",
        })

    st.success(f"총 {len(results)}개 장비에 대해 판정을 완료했습니다.")
    res_df = pd.DataFrame(results)
    st.dataframe(res_df, use_container_width=True)

    # 결과 다운로드
    csv_bytes = res_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "판정 결과 CSV 다운로드",
        data=csv_bytes,
        file_name="e26_scope_results_rec190.csv",
        mime="text/csv",
    )

if __name__ == "__main__":
    main()