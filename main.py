import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

# 페이지 기본 설정
st.set_page_config(
    page_title="서울 100년 기온 변화 분석",
    page_icon="🌡️",
    layout="wide"
)

# 데이터 로드 및 전처리 (캐싱 적용)
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/main/data/seoul.csv"
    
    # 인코딩 자동 처리
    try:
        df = pd.read_csv(url, encoding="cp949")
    except Exception:
        try:
            df = pd.read_csv(url, encoding="euc-kr")
        except Exception:
            df = pd.read_csv(url, encoding="utf-8")
    
    # 열 이름 정리 및 표준화
    df.columns = df.columns.str.strip()
    rename_dict = {}
    for col in df.columns:
        if "지점" in col:
            rename_dict[col] = "지점"
        elif "날짜" in col:
            rename_dict[col] = "날짜"
        elif "평균" in col:
            rename_dict[col] = "평균기온"
        elif "최저" in col:
            rename_dict[col] = "최저기온"
        elif "최고" in col:
            rename_dict[col] = "최고기온"
    df = df.rename(columns=rename_dict)
    
    if "지점" not in df.columns:
        df["지점"] = 108
    
    # 날짜 데이터 변환 및 숫자형 변환
    df["날짜"] = pd.to_datetime(df["날짜"])
    df["연도"] = df["날짜"].dt.year
    df["지점"] = pd.to_numeric(df["지점"], errors="coerce").fillna(108).astype(int)
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")
    df["최저기온"] = pd.to_numeric(df["최저기온"], errors="coerce")
    df["최고기온"] = pd.to_numeric(df["최고기온"], errors="coerce")
    
    # 결측치 제거
    df_clean = df.dropna(subset=["평균기온"]).copy()
    
    # 연도별/지점별 관측 일수 및 평균 산출
    yearly_df = df_clean.groupby(["연도", "지점"]).agg(
        연평균기온=("평균기온", "mean"),
        최저기온평균=("최저기온", "mean"),
        최고기온평균=("최고기온", "mean"),
        일수=("평균기온", "count")
    ).reset_index()
    
    # 관측 일수가 300일 이상인 정상 연도만 포함
    yearly_df = yearly_df[yearly_df["일수"] >= 300].copy()
    yearly_df["연평균기온"] = yearly_df["연평균기온"].round(2)
    yearly_df["최저기온평균"] = yearly_df["최저기온평균"].round(2)
    yearly_df["최고기온평균"] = yearly_df["최고기온평균"].round(2)
    
    return df, df_clean, yearly_df

# 메인 화면 타이틀
st.title("🌡️ 서울 100년 기온 변화 분석")
st.markdown("지난 100여 년간 서울의 연평균 기온 변화 추이와 원본 데이터의 데이터 특성을 요약통계로 분석합니다.")

try:
    with st.spinner("데이터를 불러오는 중입니다..."):
        raw_df, clean_df, yearly_df = load_data()

    # 사이드바 설정
    st.sidebar.header("⚙️ 분석 설정")
    
    min_year = int(yearly_df["연도"].min())
    max_year = int(yearly_df["연도"].max())
    
    year_range = st.sidebar.slider(
        "조회 연도 범위 선택",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year)
    )
    
    show_ma = st.sidebar.checkbox("10년 이동평균선 표시", value=True)
    show_trend = st.sidebar.checkbox("선형 추세선 표시", value=True)
    
    # 데이터 필터링
    filtered_yearly_df = yearly_df[(yearly_df["연도"] >= year_range[0]) & (yearly_df["연도"] <= year_range[1])].copy()
    filtered_daily_df = clean_df[(clean_df["연도"] >= year_range[0]) & (clean_df["연도"] <= year_range[1])].copy()
    
    # 결측 연도 삽입용 전체 연도 축
    full_years = pd.DataFrame({"연도": range(filtered_yearly_df["연도"].min(), filtered_yearly_df["연도"].max() + 1)})
    merged_df = pd.merge(full_years, filtered_yearly_df, on="연도", how="left")
    
    # 10년 이동평균 계산
    filtered_yearly_df["10년_이동평균"] = filtered_yearly_df["연평균기온"].rolling(window=10, min_periods=1).mean().round(2)
    merged_df = pd.merge(merged_df.drop(columns=["10년_이동평균"], errors="ignore"), 
                         filtered_yearly_df[["연도", "10년_이동평균"]], on="연도", how="left")
    
    # 이상치 및 결측 구간 계산
    missing_years = full_years[~full_years["연도"].isin(filtered_yearly_df["연도"])]
    overall_avg = filtered_yearly_df["연평균기온"].mean()
    overall_std = filtered_yearly_df["연평균기온"].std()
    low_temp_threshold = overall_avg - (overall_std * 1.5)
    outlier_low_years = filtered_yearly_df[filtered_yearly_df["연평균기온"] <= low_temp_threshold]

    # 주요 지표 (Metric)
    st.markdown("### 📊 조회 기간 핵심 지표")
    col1, col2, col3, col4 = st.columns(4)
    
    max_row = filtered_yearly_df.loc[filtered_yearly_df["연평균기온"].idxmax()]
    min_row = filtered_yearly_df.loc[filtered_yearly_df["연평균기온"].idxmin()]
    
    if len(filtered_yearly_df) >= 10:
        start_avg = filtered_yearly_df.iloc[:10]["연평균기온"].mean()
        end_avg = filtered_yearly_df.iloc[-10:]["연평균기온"].mean()
        temp_change = round(end_avg - start_avg, 2)
        change_text = f"{temp_change:+.2f} ℃"
    else:
        change_text = "N/A"

    col1.metric("최고 연평균 기온", f"{max_row['연평균기온']} ℃", f"{int(max_row['연도'])}년")
    col2.metric("최저 연평균 기온", f"{min_row['연평균기온']} ℃", f"{int(min_row['연도'])}년")
    col3.metric("조회 기간 평균 기온", f"{overall_avg:.2f} ℃")
    col4.metric("기온 변화량 (초기 10년 대비 최근 10년)", change_text)
    
    st.markdown("---")
    
    # Plotly 시각화
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=merged_df["연도"],
        y=merged_df["연평균기온"],
        mode="lines+markers",
        name="연평균 기온",
        line=dict(color="#E63946", width=2),
        marker=dict(size=5),
        connectgaps=False,
        hovertemplate="<b>%{x}년</b>: %{y}℃<extra></extra>"
    ))
    
    if show_ma:
        fig.add_trace(go.Scatter(
            x=merged_df["연도"],
            y=merged_df["10년_이동평균"],
            mode="lines",
            name="10년 이동평균",
            line=dict(color="#457B9D", width=3, dash="dash"),
            connectgaps=False,
            hovertemplate="<b>%{x}년 (10년 평균)</b>: %{y}℃<extra></extra>"
        ))
        
    if show_trend and len(filtered_yearly_df) > 1:
        z = np.polyfit(filtered_yearly_df["연도"], filtered_yearly_df["연평균기온"], 1)
        p = np.poly1d(z)
        fig.add_trace(go.Scatter(
            x=filtered_yearly_df["연도"],
            y=p(filtered_yearly_df["연도"]),
            mode="lines",
            name="선형 추세선",
            line=dict(color="#1D3557", width=2, dash="dot"),
            hovertemplate="<b>%{x}년 추세값</b>: %{y:.2f}℃<extra></extra>"
        ))

    # 결측 연도 회색 영역 표시
    for year in missing_years["연도"]:
        fig.add_vrect(
            x0=year - 0.5, x1=year + 0.5,
            fillcolor="#BDC3C7", opacity=0.3,
            layer="below", line_width=0,
            annotation_text=f"{year}", 
            annotation_position="top left",
            annotation_font=dict(color="#7F8C8D", size=10)
        )
    
    if not missing_years.empty:
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(symbol="square", color="#BDC3C7", size=10),
            name="관측 누락/부족 연도 (Break)",
            showlegend=True
        ))

    # 유난히 낮은 기온 연도 강조
    if not outlier_low_years.empty:
        fig.add_trace(go.Scatter(
            x=outlier_low_years["연도"],
            y=outlier_low_years["연평균기온"],
            mode="markers",
            name="유난히 낮은 기온 (Outlier)",
            marker=dict(
                symbol="diamond-open",
                size=12,
                color="#F1C40F",
                line=dict(width=3)
            ),
            hovertemplate="<b>%{x}년 (이상저온)</b>: %{y}℃<extra></extra>"
        ))

    fig.update_layout(
        title=dict(
            text=f"<b>서울 연평균 기온 변화 추이 및 특이 구간 ({year_range[0]}년 ~ {year_range[1]}년)</b>",
            font=dict(size=20)
        ),
        xaxis=dict(title="연도 (Year)", showgrid=True, dtick=10 if (year_range[1] - year_range[0]) > 30 else 5),
        yaxis=dict(title="평균 기온 (℃)", showgrid=True),
        hovermode="x unified",
        template="plotly_white",
        height=580,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # 지점, 평균기온, 최저기온, 최고기온 요약통계
    st.subheader("📋 원본 관측 데이터 요약통계")
    st.caption(f"선택된 조회 기간({year_range[0]}년 ~ {year_range[1]}년) 내 일별 관측 데이터의 항목별 기초 통계량입니다.")
    
    temp_cols = ["지점", "평균기온", "최저기온", "최고기온"]
    stats_df = filtered_daily_df[temp_cols].describe().T
    
    stats_df = stats_df.rename(columns={
        "count": "관측 개수(일)",
        "mean": "평균",
        "std": "표준편차",
        "min": "최소값",
        "25%": "1사분위 (25%)",
        "50%": "중앙값 (50%)",
        "75%": "3사분위 (75%)",
        "max": "최대값"
    })
    
    st.dataframe(
        stats_df.style.format({
            "관측 개수(일)": "{:,.0f}",
            "평균": "{:.2f}",
            "표준편차": "{:.2f}",
            "최소값": "{:.1f}",
            "1사분위 (25%)": "{:.1f}",
            "중앙값 (50%)": "{:.1f}",
            "3사분위 (75%)": "{:.1f}",
            "최대값": "{:.1f}"
        }),
        use_container_width=True
    )
    
    # 지점이 포함된 연도별 상세 집계 표
    with st.expander("📊 연도별 집계 데이터 상세 보기"):
        def highlight_outliers(s):
            is_low = s.name == "연평균기온" and s <= low_temp_threshold
            return ['background-color: #FEF9E7' if v else '' for v in is_low]

        display_yearly = filtered_yearly_df[["지점", "연도", "연평균기온", "최저기온평균", "최고기온평균", "10년_이동평균"]]
        st.dataframe(
            display_yearly.style.format({
                "지점": "{:.0f}",
                "연평균기온": "{:.2f} ℃",
                "최저기온평균": "{:.2f} ℃",
                "최고기온평균": "{:.2f} ℃",
                "10년_이동평균": "{:.2f} ℃"
            }).apply(highlight_outliers, axis=1),
            use_container_width=True
        )

except Exception as e:
    st.error(f"데이터를 불러오거나 처리하는 중 오류가 발생했습니다: {e}")
