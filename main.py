pip install -r requirements.txt
streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from io import StringIO
from typing import List, Tuple

st.set_page_config(page_title="국가별 MBTI 분포 뷰어", page_icon="🧭", layout="wide")

# -----------------------------
# 유틸함수
# -----------------------------
ENCODINGS_TO_TRY = ["utf-8", "utf-8-sig", "cp949", "euc-kr"]

MBTI_TYPES = [
    "INTJ","INTP","ENTJ","ENTP","INFJ","INFP","ENFJ","ENFP",
    "ISTJ","ISFJ","ESTJ","ESFJ","ISTP","ISFP","ESTP","ESFP"
]

def detect_mbti_columns(df: pd.DataFrame) -> List[str]:
    """컬럼명에서 MBTI 16유형 컬럼 자동 탐지"""
    mbti_cols = []
    for col in df.columns:
        norm = str(col).strip().lower().replace("%","").replace(" ","").replace("_","")
        for t in MBTI_TYPES:
            if norm == t.lower() or norm.startswith(t.lower()):
                mbti_cols.append(col)
                break
    # 순서 보존 중복 제거
    seen = set()
    mbti_cols = [c for c in mbti_cols if not (c in seen or seen.add(c))]
    return mbti_cols

def to_numeric_clean(s: pd.Series) -> pd.Series:
    """문자열 %/쉼표 제거 후 숫자 변환"""
    if s.dtype.kind in "biufc":
        return s
    cleaned = (
        s.astype(str)
         .str.replace("%","",regex=False)
         .str.replace(",","",regex=False)
         .replace({"nan": np.nan})
    )
    return pd.to_numeric(cleaned, errors="coerce")

def normalize_unit_to_percent(df: pd.DataFrame, mbti_cols: List[str]) -> Tuple[pd.DataFrame, str]:
    """
    MBTI 합계 평균을 보고 0~1(비율)인지 0~100(퍼센트)인지 판별.
    모두 '퍼센트(0~100)'로 변환하여 반환.
    """
    tmp = df.copy()
    for c in mbti_cols:
        tmp[c] = to_numeric_clean(tmp[c])
    tmp["__sum__"] = tmp[mbti_cols].sum(axis=1, skipna=True)
    mean_sum = tmp["__sum__"].mean()

    if 0.95 <= mean_sum <= 1.05:       # 비율
        for c in mbti_cols:
            tmp[c] = tmp[c] * 100.0
        unit = "비율(0~1) → 퍼센트(0~100)로 변환"
    else:
        unit = "퍼센트(0~100) 그대로 사용"

    tmp.drop(columns=["__sum__"], inplace=True)
    return tmp, unit

def robust_read_csv(uploaded_file_or_path) -> Tuple[pd.DataFrame, str]:
    """여러 인코딩 시도 로더"""
    last_err = None
    for enc in ENCODINGS_TO_TRY:
        try:
            df = pd.read_csv(uploaded_file_or_path, encoding=enc)
            return df, enc
        except Exception as e:
            last_err = e
    raise RuntimeError(f"CSV를 읽지 못했습니다. 마지막 오류: {last_err}")

@st.cache_data(show_spinner=False)
def load_data(file_bytes: bytes | None, local_path: str | None) -> Tuple[pd.DataFrame, str]:
    """업로드 파일 우선, 없으면 로컬 경로 시도"""
    if file_bytes is not None:
        # 업로드 파일은 바이트 → StringIO로 변환하여 인코딩 탐지
        content = file_bytes.decode("utf-8", errors="ignore")
        # 먼저 utf-8 시도, 실패 시 다양한 인코딩 재시도
        try:
            return pd.read_csv(StringIO(content)), "utf-8*"
        except Exception:
            pass
        # 재시도
        return robust_read_csv(StringIO(content))
    if local_path:
        return robust_read_csv(local_path)
    raise RuntimeError("데이터 소스가 없습니다. CSV를 업로드하거나 경로를 지정하세요.")

def guess_country_col(df: pd.DataFrame, mbti_cols: List[str]) -> str | None:
    """국가/식별자 컬럼 추정: MBTI 컬럼이 아닌 것 중 우선순위로 탐색"""
    candidates = [c for c in df.columns if c not in mbti_cols]
    # 우선순위: Country, country, 국가, Nation, Name 등
    priority = ["Country","country","국가","Nation","nation","Name","name","지역","지역명"]
    for p in priority:
        if p in df.columns and p not in mbti_cols:
            return p
    return candidates[0] if candidates else None

def tidy_country_row(row: pd.Series, mbti_cols: List[str]) -> pd.DataFrame:
    """한 국가 행을 세로형(Type, Percent)으로 변환하고 내림차순 정렬"""
    data = []
    for c in mbti_cols:
        val = to_numeric_clean(pd.Series([row[c]])).iloc[0]
        data.append((c, val))
    out = pd.DataFrame(data, columns=["Type", "Percent"])
    out = out.sort_values("Percent", ascending=False).reset_index(drop=True)
    return out

# -----------------------------
# 사이드바 & 데이터 로드
# -----------------------------
st.title("🧭 국가별 MBTI 분포 뷰어")
st.caption("CSV에서 국가를 선택하면, 해당 국가의 MBTI 16유형 비율을 내림차순으로 보여줍니다. (단위 자동 인식)")

with st.sidebar:
    st.header("📂 데이터 소스")
    uploaded = st.file_uploader("MBTI CSV 업로드", type=["csv"])
    use_local = st.toggle("로컬 기본 경로 사용(있을 때)", value=True)
    local_path = "countriesMBTI_16types.csv" if use_local else None
    st.markdown("- 컬럼 예시: `Country`, `INFJ`, `ISFJ`, ..., `ESFJ`")

try:
    df, used_encoding = load_data(uploaded.getvalue() if uploaded else None, local_path)
except Exception as e:
    st.error(f"데이터 로드 실패: {e}")
    st.stop()

mbti_cols = detect_mbti_columns(df)
if len(mbti_cols) != 16:
    st.warning(f"MBTI 컬럼 감지 수: {len(mbti_cols)} (예상 16). 컬럼명을 확인해 주세요.\n감지됨: {mbti_cols}")

df_norm, unit_info = normalize_unit_to_percent(df, mbti_cols)
country_col = guess_country_col(df_norm, mbti_cols)

col_l, col_r = st.columns([2, 1])
with col_l:
    st.subheader("데이터 개요")
    st.write(f"- 행/열: **{df_norm.shape[0]} x {df_norm.shape[1]}**, 인코딩: **{used_encoding}**")
    st.write(f"- MBTI 컬럼(16): {', '.join(mbti_cols)}")
    st.write(f"- 단위 처리: **{unit_info}** → 시각화/표시는 **퍼센트(%)** 기준")
with col_r:
    st.dataframe(df_norm.head(10), use_container_width=True)

if not country_col:
    st.error("국가(식별자) 컬럼을 찾을 수 없습니다. 'Country' 또는 국가명을 담은 컬럼이 필요합니다.")
    st.stop()

# 국가 선택
countries = df_norm[country_col].astype(str).dropna().unique().tolist()
countries.sort()
selected = st.selectbox("국가 선택", options=countries, index=0)

# 선택 행 추출
row = df_norm.loc[df_norm[country_col].astype(str) == str(selected)]
if row.empty:
    st.error("선택한 국가의 데이터가 없습니다.")
    st.stop()

row = row.iloc[0]
tidy_df = tidy_country_row(row, mbti_cols)

# 합계/오차 안내
sum_percent = float(np.nansum(tidy_df["Percent"].values))
warn = ""
if not (99.0 <= sum_percent <= 101.0):
    warn = f"⚠️ 합계가 {sum_percent:.2f}% 입니다(이상치일 수 있음)."
else:
    warn = f"✅ 합계: {sum_percent:.2f}% (정상 범위)"

st.markdown(f"### 🇺🇳 {selected} — MBTI 분포 (내림차순)")
st.caption(warn)

# 시각화
chart = (
    alt.Chart(tidy_df)
    .mark_bar()
    .encode(
        x=alt.X("Percent:Q", title="비율(%)"),
        y=alt.Y("Type:N", sort="-x", title="MBTI 유형"),
        tooltip=[alt.Tooltip("Type:N", title="유형"),
                 alt.Tooltip("Percent:Q", title="비율(%)", format=".2f")]
    )
    .properties(height=420)
)

st.altair_chart(chart, use_container_width=True)

# 표 (퍼센트 포맷)
fmt_df = tidy_df.copy()
fmt_df["비율(%)"] = fmt_df["Percent"].map(lambda v: f"{v:.2f}")
fmt_df = fmt_df[["Type","비율(%)"]]
st.dataframe(fmt_df, use_container_width=True, hide_index=True)

# 다운로드 버튼
csv_bytes = tidy_df.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    label="⬇️ 이 국가의 MBTI 분포 CSV 다운로드",
    data=csv_bytes,
    file_name=f"{selected}_MBTI_distribution.csv",
    mime="text/csv"
)

# 재미 요소(선택 시 1회)
if st.button("🎈 축하 풍선 한번!"):
    st.balloons()
