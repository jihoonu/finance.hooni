import streamlit as st
import pandas as pd
import datetime 
from analyzer import run_financial_analysis

st.set_page_config(page_title="모바일 재무분석기", page_icon="📈", layout="wide")

st.title("📈 재무제표 분석 프로그램")
st.markdown("시작 연도만 입력하면 최근 실적까지 원클릭 자동 분석")

st.sidebar.header("설정")
api_key = st.sidebar.text_input("DART API 키를 입력하세요", type="password")
company_name = st.sidebar.text_input("기업명", "삼성전자")

st.sidebar.divider()
current_year = datetime.datetime.now().year
start_year = st.sidebar.number_input("시작 연도 입력", min_value=2010, max_value=current_year, value=2022, step=1)
st.sidebar.caption(f"※ {start_year}년부터 공시된 가장 최근 보고서까지 자동으로 수집합니다.")

def format_by_index(col):
    formatted_col = []
    for idx, val in col.items():
        if any(kw in str(idx) for kw in ['%', '회전율', '회전기간', '비율']):
            formatted_col.append(f"{val:,.2f}")
        else:
            formatted_col.append(f"{val:,.0f}")
    return formatted_col

if st.sidebar.button("📊 재무 데이터 자동 수집"):
    if not api_key:
        st.warning("API 키를 먼저 입력해주세요.")
    else:
        progress_bar = st.progress(0, text="데이터 수집 준비 중...")

        def update_progress(current, total, message):
            progress_bar.progress(current / total, text=message)

        result_df = run_financial_analysis(
            api_key=api_key, 
            company_name=company_name, 
            start_year=start_year, 
            progress_callback=update_progress
        )

        progress_bar.empty()

        if result_df is None or result_df.empty:
            st.error("해당 기간의 데이터를 불러오지 못했습니다. 기업명이나 연도를 확인하세요.")
        else:
            st.success("데이터 수집 완료!")
            st.session_state['analyzed_data'] = result_df
            st.session_state['company'] = company_name

if 'analyzed_data' in st.session_state:
    result_df = st.session_state['analyzed_data']
    comp_name = st.session_state['company']

    st.subheader(f"📊 {comp_name} 요약 재무분석표")

    view_mode = st.radio("보기 방식 선택", ["전체 분기 연속 보기 (디테일)", "연간 실적만 보기 (요약)"], horizontal=True)

    st.markdown("<div style='text-align: right; color: gray; font-size: 14px;'>[단위: 백만 원, %, 회, 일]</div>", unsafe_allow_html=True)

    # 선택된 뷰 모드에 따라 데이터프레임 필터링
    if view_mode == "연간 실적만 보기 (요약)":
        annual_cols = [c for c in result_df.columns if '4Q' in c]
        if not annual_cols:
            st.warning("아직 4분기(연간) 데이터가 조회되지 않았습니다.")
            display_df = result_df.copy()
        else:
            display_df = result_df[annual_cols].copy()
    else:
        display_df = result_df.copy()

    # 1. 표 출력 (3개 탭으로 분리)
    formatted_df = display_df.apply(format_by_index)

    tab_bs, tab_is, tab_cf = st.tabs(["🏛️ 재무상태표", "📊 손익계산서", "💸 현금흐름표"])

    with tab_bs:
        # 재무상태표: 0~9번째 줄
        st.dataframe(formatted_df.iloc[0:10], use_container_width=True)

    with tab_is:
        # 손익계산서: 10~27번째 줄 (분석기 로직 수정에 맞춰 길이를 28로 변경)
        st.dataframe(formatted_df.iloc[10:28], use_container_width=True)

    with tab_cf:
        # 현금흐름표: 28번째 줄부터 끝까지
        st.dataframe(formatted_df.iloc[28:], use_container_width=True)

    st.divider()

    # ==========================================
    # 💡 [그래프 섹션] 시각화 차트 그리기
    # ==========================================
    st.subheader("📉 핵심 지표 추이 시각화")

    # 차트를 그리기 위해 행(분석항목)과 열(기간)을 뒤집습니다 (Transpose).
    chart_df = display_df.T

    # 인덱스 이름(항목명)을 정확히 찾아 매칭
    col_revenue = [c for c in chart_df.columns if '11. 매출액' in c][0]
    col_op_income = [c for c in chart_df.columns if '14. 영업이익' in c and '%' not in c][0]
    col_net_income = [c for c in chart_df.columns if '17. 당기순이익' in c][0]

    col_roe = [c for c in chart_df.columns if '18. ROE' in c][0]
    col_roic = [c for c in chart_df.columns if '19. ROIC' in c][0]

    col_debt_ratio = [c for c in chart_df.columns if '10. 부채비율' in c][0]

    # 모바일에서 보기 좋게 탭(Tab)으로 나누어 배치
    tab1, tab2, tab3 = st.tabs(["💰 실적 추이", "🔥 수익성 (ROE/ROIC)", "⚖️ 부채비율"])

    with tab1:
        st.markdown("**매출액, 영업이익, 당기순이익 추이 (단위: 백만 원)**")
        # 매출액과 이익들은 막대그래프(Bar Chart)로 출력
        st.bar_chart(chart_df[[col_revenue, col_op_income, col_net_income]], use_container_width=True)

    with tab2:
        st.markdown("**ROE 및 ROIC 추이 (단위: %)**")
        # 수익률은 꺾은선그래프(Line Chart)로 출력
        st.line_chart(chart_df[[col_roe, col_roic]], use_container_width=True)

    with tab3:
        st.markdown("**부채비율 추이 (단위: %)**")
        st.line_chart(chart_df[[col_debt_ratio]], color="#ff4b4b", use_container_width=True)
