import re
import streamlit as st

from e26_classifier import EquipmentInput, E26Classifier


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

    st.title("IACS UR E26 대상 장비 판정 도우미 (텍스트 붙여넣기 버전)")
    st.write(
        """
        - 엑셀에서 행들을 복사해서 그대로 붙여넣어도 되고,  
        - 메일/문서에 있는 장비 리스트를 줄 단위로 붙여넣어도 됩니다.  
        - 각 **줄 = 장비 1개**로 간주해서 UR E26 대상 여부를 판정합니다.
        """
    )

    sample_help = (
        "예시:\n"
        "WATER INGRESS ALARM SYSTEM\tEMBEDDED CONTROLLER\tCPU CORE MODULE\tCOMFILE TECHNOLOGY\tALARM MONITORING\tWheel house\tCUBLOC STUDIO\t3.0C\n"
        "Anchoring and mooring\tDECK MACHINERY\tElectric deck machinery system\tFlutek\t...\n"
    )

    raw_text = st.text_area(
        "장비 목록 텍스트를 여기에 붙여넣으세요 (한 줄당 장비 1개)",
        height=300,
        placeholder=sample_help
    )

    col_run, col_clear = st.columns([1, 1])
    with col_clear:
        if st.button("입력 지우기"):
            # Streamlit은 상태 클리어를 바로 해주진 않아서, 그냥 빈 문자열을 다시 세팅하는 느낌으로 사용하면 됨
            st.experimental_rerun()

    with col_run:
        run_clicked = st.button("UR E26 판정 실행")

    if not run_clicked or not raw_text.strip():
        return

    # --- 분류 시작 ---
    clf = E26Classifier()
    items = parse_lines(raw_text)

    if not items:
        st.warning("유효한 줄이 없습니다. 최소 한 줄 이상의 장비 정보를 붙여넣어 주세요.")
        return

    results = []
    for item in items:
        eq = build_equipment_input(item)
        decision = clf.classify(eq)

        results.append({
            "라인": item["id"],
            "장비/시스템 이름(추출)": item["system_name"],
            "E26 대상 여부": "대상" if decision.in_scope else "비대상",
            "CBS 여부": "CBS" if decision.is_cbs else "비CBS",
            "Function Class": decision.function_class or "",
            "Score": f"{decision.score:.2f}",
            "Confidence": f"{decision.confidence:.2f}",
            "Warnings": " | ".join(decision.warnings) if decision.warnings else "",
            "설명": decision.explanation,
        })

    st.success(f"총 {len(results)}개 장비에 대해 판정을 완료했습니다.")

    # 요약 테이블 (대상/비대상만 간단히 보고 싶을 때)
    st.subheader("요약 결과 (라인별 E26 대상 여부)")
    summary_rows = [
        {
            "라인": r["라인"],
            "장비/시스템 이름": r["장비/시스템 이름(추출)"],
            "E26 대상 여부": r["E26 대상 여부"],
            "CBS 여부": r["CBS 여부"],
            "Function Class": r["Function Class"],
        }
        for r in results
    ]
    st.dataframe(summary_rows, use_container_width=True)

    # 상세 설명 토글
    st.subheader("상세 결과 및 판단 근거")
    for r in results:
        with st.expander(f"[라인 {r['라인']}] {r['장비/시스템 이름(추출)']} ({r['E26 대상 여부']}, {r['CBS 여부']})"):
            st.markdown(f"**Function Class:** `{r['Function Class']}`")
            st.markdown(f"**Score:** `{r['Score']}`, **Confidence:** `{r['Confidence']}`")
            if r["Warnings"]:
                st.markdown(f"**Warnings:** {r['Warnings']}")
            st.markdown("**판단 설명 로그:**")
            st.code(r["설명"], language="text")


if __name__ == "__main__":
    main()
