import io
import re

import numpy as np
import pandas as pd
import streamlit as st

from e26_classifier import EquipmentInput, E26Classifier


HEADER_KEYWORDS = [
    "system name", "system", "component", "manufacturer", "model",
    "function", "installed location", "external communication",
    "ext.", "no.", "index"
]


def is_header_like(text: str, row_index: int) -> bool:
    """
    샘플 형식 기준으로 '제목/헤더/설명' 행을 최대한 잘 걸러내기 위한 휴리스틱.
    - 파일 상단부(0~1행)는 거의 항상 제목/프로젝트 정보 → 무조건 헤더 취급
    - 'List of Hardware Components', 'List of Software Components' 같은 문구
    - 'System Name / Component Name / Installed Location / IP/Subnet ...' 등 컬럼 헤더
    """
    if not text:
        return False

    t = text.lower()

    # 1) 맨 위 0~1행은 프로젝트 제목/문서 제목일 확률이 높으니 바로 헤더 처리
    if row_index <= 1:
        return True

    # 2) 섹션 타이틀들
    phrases = [
        "list of hardware components",
        "list of software components",
        "system info.",
        "hardware info.",
        "network info.",
        "software info.",
    ]
    if any(p in t for p in phrases):
        return True

    # 3) 컬럼 헤더 패턴
    col_phrases = [
        "system\nname", "system name",
        "component name",
        "manufacturer",
        "function/purpose", "function / purpose", "function /\npurpose",
        "installed location",
        "system software", "name/type", "version",
        "ip/subnet", "connected cbs",
        "remote\nconnection", "remote connection",
        "supported\nprotocol", "supported protocol",
        "lan", "usb",
        "no.",
    ]
    hit = sum(1 for kw in col_phrases if kw in t)

    # 이런 키워드가 2개 이상 들어가 있으면 거의 100% 컬럼 헤더
    if hit >= 2:
        return True

    return False

def row_to_text(row: pd.Series) -> str:
    """
    한 행(row)의 모든 셀을 문자열로 합쳐서 하나의 라인 텍스트로 만든다.
    NaN / 빈 셀은 무시.
    """
    parts = []
    for v in row.values:
        if pd.isna(v):
            continue
        s = str(v).strip()
        if not s or s.lower() == "nan":
            continue
        parts.append(s)
    return " | ".join(parts)


import re

def extract_system_name(text: str) -> str:
    """
    한 행 전체 텍스트에서 '장비/시스템 이름' 후보를 추출.
    - 첫 번째 토큰이 '1', '2' 같은 번호거나 'No.' 이면 건너뛰고
    - 다음 토큰부터 유의미한 이름을 찾는다.
    """
    if not text:
        return ""

    parts = re.split(r"[\t,;|]", text)
    for p in parts:
        name = p.strip()
        if not name:
            continue

        lower = name.lower()

        # 순번(1, 2, 3...)은 system name 아님
        if re.fullmatch(r"\d+", name):
            continue
        # 'No', 'No.'도 무시
        if lower in ("no", "no."):
            continue

        return name[:80]  # 너무 길면 표시만 잘라줌

    # 못 찾으면 전체 텍스트 일부라도 반환
    return text[:80]


def build_equipment_input_from_row(row_text: str, system_name: str, id_str: str) -> EquipmentInput:
    """
    한 행 텍스트 → E26Classifier에서 쓰는 EquipmentInput 객체로 변환.
    - system/name: system_name
    - description: row_text 전체
    나머지는 정보 부족 → None/False로 두고 classifier 휴리스틱에 맡김.
    """
    eq = EquipmentInput(
        id=id_str,
        name=system_name,
        vendor="",
        system=system_name,
        function_hint=None,
        description=row_text,
        is_computer_based=None,
        controls_physical_process=None,
        network_zone=None,
        connected_to_external_network=False,
    )
    return eq


from collections import defaultdict

def classify_dataframe(df: pd.DataFrame, clf: E26Classifier, sheet_name: str):
    """
    하나의 DataFrame(시트 하나)에 대해:
    1) 각 행을 텍스트로 만들고
    2) 헤더/설명 행은 버리고
    3) system_name을 뽑아서,
    4) 같은 system_name끼리 묶어 한 번만 판정한다.
    """
    # 1단계: 후보 행 뽑기
    candidates = []  # [{sheet, row_index, system_name, row_text}, ...]

    for idx, row in df.iterrows():
        row_text = row_to_text(row)
        if not row_text:
            continue

        # 헤더/설명/컬럼 행이면 스킵
        if is_header_like(row_text, idx):
            continue

        system_name = extract_system_name(row_text)
        if not system_name:
            continue

        candidates.append({
            "sheet": sheet_name,
            "row_index": idx + 1,  # 사람 눈에 보이는 행 번호(1부터)
            "system_name": system_name,
            "row_text": row_text,
        })

    if not candidates:
        return []

    # 2단계: 같은 system_name끼리 그룹핑
    grouped = defaultdict(list)
    for item in candidates:
        key = (item["sheet"], item["system_name"])
        grouped[key].append(item)

    # 3단계: 그룹 단위로 classifier 호출
    results = []

    for (sheet, system_name), items in grouped.items():
        # 같은 system에 속한 여러 행의 텍스트를 하나로 합친다
        combined_text = " || ".join(i["row_text"] for i in items)
        first_row = min(i["row_index"] for i in items)
        row_indices = sorted(i["row_index"] for i in items)

        eq_id = f"{sheet}-{first_row}"
        eq = build_equipment_input_from_row(combined_text, system_name, eq_id)
        decision = clf.classify(eq)

        results.append({
            "sheet": sheet,
            "row_index": first_row,          # 대표 행 번호
            "row_indices": row_indices,      # 실제로 묶인 행들
            "system_name": system_name,
            "row_text": combined_text,
            "in_scope": decision.in_scope,
            "is_cbs": decision.is_cbs,
            "function_class": decision.function_class or "",
            "score": decision.score,
            "confidence": decision.confidence,
            "warnings": decision.warnings,
            "explanation": decision.explanation,
        })

    return results



def main():
    st.set_page_config(page_title="UR E26 대상 장비 판정 (파일 업로드 버전)", layout="wide")

    st.title("IACS UR E26 대상 장비 판정 도우미 – 파일 업로드 버전")
    st.write(
        """
        - Yard에서 받은 **엑셀(또는 CSV) 파일**을 그대로 업로드하면,  
          각 시트/행 텍스트를 모두 읽어서 UR E26 대상 장비 여부를 판정합니다.  
        - 특정 형식(Sample 형식)에 딱 맞을 필요 없이,  
          **표 형태로만 되어 있으면 웬만큼은 알아서 읽습니다.**
        """
    )

    uploaded_file = st.file_uploader("엑셀 또는 CSV 파일을 선택하세요", type=["xlsx", "xls", "csv"])

    if not uploaded_file:
        st.info("좌측 또는 위의 파일 선택 버튼을 통해 파일을 업로드해주세요.")
        return

    # 분류기 준비
    clf = E26Classifier()

    # 파일 읽기
    dfs = {}
    try:
        if uploaded_file.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded_file, header=None)
            dfs["Sheet1"] = df
        else:
            # 엑셀의 모든 시트를 header=None으로 읽어서 '있는 그대로' DataFrame으로 받는다
            xls = pd.ExcelFile(uploaded_file)
            for sheet_name in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=sheet_name, header=None)
                dfs[sheet_name] = df
    except Exception as e:
        st.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")
        return

    all_results = []

    for sheet_name, df in dfs.items():
        sheet_results = classify_dataframe(df, clf, sheet_name)
        all_results.extend(sheet_results)

    if not all_results:
        st.warning("장비로 인식할 수 있는 행을 찾지 못했습니다. 표 구조인지, 혹은 너무 단순한 텍스트인지 확인해 주세요.")
        return

    # DataFrame으로 변환
    res_df = pd.DataFrame(all_results)

    # 요약: 대상/비대상 개수
    num_total = len(res_df)
    num_in_scope = int(res_df["in_scope"].sum())
    num_out_scope = num_total - num_in_scope

    st.success(f"총 {num_total}개 행(장비 후보)을 분석했습니다.")
    st.write(f"- UR E26 **대상으로 판정된 장비**: {num_in_scope}개")
    st.write(f"- UR E26 **비대상으로 판정된 장비**: {num_out_scope}개")

    # 필터 옵션
    st.subheader("결과 필터")
    col1, col2, col3 = st.columns(3)
    with col1:
        show_only_in_scope = st.checkbox("E26 대상 장비만 보기", value=True)
    with col2:
        min_conf = st.slider("최소 Confidence (Function 분류 신뢰도)", 0.0, 1.0, 0.0, 0.05)
    with col3:
        show_only_cbs = st.checkbox("CBS로 판단된 장비만 보기", value=False)

    filtered = res_df.copy()
    if show_only_in_scope:
        filtered = filtered[filtered["in_scope"] == True]
    if show_only_cbs:
        filtered = filtered[filtered["is_cbs"] == True]
    if min_conf > 0.0:
        filtered = filtered[filtered["confidence"] >= min_conf]

    if filtered.empty:
        st.warning("현재 필터 조건에 해당하는 장비가 없습니다. 필터를 조정해보세요.")
        return

    # 요약 테이블
    st.subheader("요약 결과 (행별 E26 대상 여부)")
    summary_cols = filtered[[
        "sheet", "row_index", "system_name", "in_scope", "is_cbs", "function_class", "score", "confidence"
    ]].copy()
    summary_cols["E26 대상 여부"] = summary_cols["in_scope"].map({True: "대상", False: "비대상"})
    summary_cols["CBS 여부"] = summary_cols["is_cbs"].map({True: "CBS", False: "비CBS"})
    summary_cols = summary_cols.drop(columns=["in_scope", "is_cbs"])

    st.dataframe(summary_cols, use_container_width=True)

    # 상세 설명
    st.subheader("상세 결과 및 판단 근거")
    for _, r in filtered.iterrows():
        title = f"[{r['sheet']} / {int(r['row_index'])}행] {r['system_name']} – { '대상' if r['in_scope'] else '비대상' } / { 'CBS' if r['is_cbs'] else '비CBS' }"
        with st.expander(title):
            st.markdown(f"**Function Class:** `{r['function_class']}`")
            st.markdown(f"**Score:** `{r['score']:.2f}`, **Confidence:** `{r['confidence']:.2f}`")

            if r["warnings"]:
                st.markdown(f"**Warnings:** { ' | '.join(r['warnings']) if isinstance(r['warnings'], list) else r['warnings'] }")

            st.markdown("**원본 행 텍스트:**")
            st.code(r["row_text"], language="text")

            st.markdown("**판단 설명 로그:**")
            st.code(r["explanation"], language="text")


if __name__ == "__main__":
    main()
