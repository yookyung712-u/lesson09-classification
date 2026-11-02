from pathlib import Path
import zipfile

page_code = r'''import pandas as pd
import plotly.express as px
import streamlit as st

# 페이지 설정은 앱의 첫 페이지(main.py)에서 이미 적용됩니다.
st.title("🔎 탐색")
st.caption("뇌졸중 데이터의 분포와 그룹별 차이를 그래프로 살펴봅니다.")

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"


@st.cache_data(ttl=3600)
def load_data():
    """첫 화면과 같은 주소에서 CSV를 UTF-8로 읽습니다."""
    return pd.read_csv(DATA_URL, encoding="utf-8")


try:
    data = load_data()
except Exception as exc:
    st.error("데이터를 불러오지 못했습니다. 인터넷 연결과 데이터 주소를 확인해 주세요.")
    st.code(str(exc))
    st.stop()

required_columns = [
    "age", "avg_glucose_level", "hypertension",
    "heart_disease", "bmi", "smoking_status", "stroke",
]
missing_columns = [column for column in required_columns if column not in data.columns]
if missing_columns:
    st.error(f"분석에 필요한 열이 없습니다: {', '.join(missing_columns)}")
    st.stop()

# 분석에 사용하는 열은 숫자로 변환하고, 변환할 수 없는 값은 결측 처리합니다.
data = data.copy()
for column in ["age", "avg_glucose_level", "hypertension", "heart_disease", "bmi", "stroke"]:
    data[column] = pd.to_numeric(data[column], errors="coerce")

data["뇌졸중 구분"] = data["stroke"].map({0: "겪지 않음 (stroke=0)", 1: "겪음 (stroke=1)"})
stroke_order = ["겪지 않음 (stroke=0)", "겪음 (stroke=1)"]

# --------------------------------------------------
# 1. 나이와 평균 혈당 분포
# --------------------------------------------------
st.header("1. 나이와 평균 혈당의 분포")
hist_left, hist_right = st.columns(2)

with hist_left:
    age_values = data["age"].dropna()
    fig_age = px.histogram(
        age_values.to_frame(name="나이"),
        x="나이",
        nbins=30,
        title="나이 분포",
        labels={"나이": "나이"},
    )
    fig_age.update_layout(yaxis_title="사람 수", xaxis_title="나이", bargap=0.06)
    st.plotly_chart(fig_age, use_container_width=True)

with hist_right:
    glucose_values = data["avg_glucose_level"].dropna()
    fig_glucose = px.histogram(
        glucose_values.to_frame(name="평균 혈당"),
        x="평균 혈당",
        nbins=30,
        title="평균 혈당 분포",
        labels={"평균 혈당": "평균 혈당"},
    )
    fig_glucose.update_layout(yaxis_title="사람 수", xaxis_title="평균 혈당", bargap=0.06)
    st.plotly_chart(fig_glucose, use_container_width=True)

st.caption("각 막대는 해당 구간에 속하는 사람 수를 나타냅니다. 빈 값은 분포 그래프에서 제외했습니다.")

# --------------------------------------------------
# 2. 뇌졸중 그룹별 상자그림 및 평균표
# --------------------------------------------------
st.header("2. 뇌졸중을 겪은 사람과 겪지 않은 사람 비교")
comparison_data = data[data["뇌졸중 구분"].notna()].copy()

box_left, box_right = st.columns(2)
with box_left:
    age_box_data = comparison_data[comparison_data["age"].notna()]
    fig_age_box = px.box(
        age_box_data,
        x="뇌졸중 구분",
        y="age",
        category_orders={"뇌졸중 구분": stroke_order},
        title="뇌졸중 구분에 따른 나이",
        labels={"뇌졸중 구분": "그룹", "age": "나이"},
        points=False,
    )
    st.plotly_chart(fig_age_box, use_container_width=True)

with box_right:
    glucose_box_data = comparison_data[comparison_data["avg_glucose_level"].notna()]
    fig_glucose_box = px.box(
        glucose_box_data,
        x="뇌졸중 구분",
        y="avg_glucose_level",
        category_orders={"뇌졸중 구분": stroke_order},
        title="뇌졸중 구분에 따른 평균 혈당",
        labels={"뇌졸중 구분": "그룹", "avg_glucose_level": "평균 혈당"},
        points=False,
    )
    st.plotly_chart(fig_glucose_box, use_container_width=True)

mean_table = (
    comparison_data.groupby("뇌졸중 구분", observed=False)
    .agg(
        사람_수=("stroke", "size"),
        평균_나이=("age", "mean"),
        평균_혈당=("avg_glucose_level", "mean"),
    )
    .reindex(stroke_order)
    .rename_axis("그룹")
    .reset_index()
)
mean_table = mean_table.rename(columns={
    "사람_수": "사람 수",
    "평균_나이": "평균 나이",
    "평균_혈당": "평균 혈당",
})
mean_table["평균 나이"] = mean_table["평균 나이"].round(2)
mean_table["평균 혈당"] = mean_table["평균 혈당"].round(2)
st.subheader("그룹별 평균값")
st.dataframe(mean_table, use_container_width=True, hide_index=True)
st.caption("평균은 각 항목의 빈 값을 제외하고 계산합니다. 상자그림은 중앙값과 사분위 범위를 보여 줍니다.")

# --------------------------------------------------
# 3. 고혈압 및 심장병 여부에 따른 뇌졸중 비율
# --------------------------------------------------
st.header("3. 고혈압·심장병 여부에 따른 뇌졸중 비율")


def make_rate_table(column, display_name):
    """0/1 범주별로 stroke=1 비율과 분모를 계산합니다."""
    subset = data[data[column].isin([0, 1]) & data["stroke"].isin([0, 1])].copy()
    grouped = subset.groupby(column, observed=False).agg(
        전체_사람=("stroke", "size"),
        뇌졸중_사람=("stroke", lambda values: int((values == 1).sum())),
    )
    grouped = grouped.reindex([0, 1]).rename(index={0: "없음 (0)", 1: "있음 (1)"})
    grouped["뇌졸중_비율"] = grouped["뇌졸중_사람"] / grouped["전체_사람"] * 100
    result = grouped.reset_index(names=display_name)
    result = result.rename(columns={
        "전체_사람": "전체 사람 수",
        "뇌졸중_사람": "뇌졸중을 겪은 사람 수",
        "뇌졸중_비율": "뇌졸중 비율(%)",
    })
    return result


rate_left, rate_right = st.columns(2)
with rate_left:
    hypertension_rates = make_rate_table("hypertension", "고혈압")
    fig_hypertension = px.bar(
        hypertension_rates,
        x="고혈압",
        y="뇌졸중 비율(%)",
        text="뇌졸중 비율(%)",
        title="고혈압 여부별 뇌졸중 비율",
        labels={"고혈압": "고혈압", "뇌졸중 비율(%)": "뇌졸중 비율 (%)"},
    )
    fig_hypertension.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    fig_hypertension.update_layout(yaxis_ticksuffix="%", yaxis_rangemode="tozero")
    st.plotly_chart(fig_hypertension, use_container_width=True)
    st.dataframe(hypertension_rates, use_container_width=True, hide_index=True)

with rate_right:
    heart_rates = make_rate_table("heart_disease", "심장병")
    fig_heart = px.bar(
        heart_rates,
        x="심장병",
        y="뇌졸중 비율(%)",
        text="뇌졸중 비율(%)",
        title="심장병 여부별 뇌졸중 비율",
        labels={"심장병": "심장병", "뇌졸중 비율(%)": "뇌졸중 비율 (%)"},
    )
    fig_heart.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    fig_heart.update_layout(yaxis_ticksuffix="%", yaxis_rangemode="tozero")
    st.plotly_chart(fig_heart, use_container_width=True)
    st.dataframe(heart_rates, use_container_width=True, hide_index=True)

st.caption("각 비율은 해당 조건에 속한 사람 중 stroke=1인 사람의 비율입니다. 이 비교만으로 원인이나 인과관계를 판단할 수는 없습니다.")

# --------------------------------------------------
# 4. BMI 결측 그룹과 전체의 뇌졸중 비율
# --------------------------------------------------
st.header("4. BMI가 비어 있는 사람들의 뇌졸중 비율")
stroke_valid = data[data["stroke"].isin([0, 1])].copy()
missing_bmi_group = stroke_valid[stroke_valid["bmi"].isna()]
comparison_groups = [
    ("BMI가 비어 있는 사람", missing_bmi_group),
    ("전체 사람", stroke_valid),
]

bmi_summary_rows = []
for group_name, group in comparison_groups:
    count = len(group)
    cases = int((group["stroke"] == 1).sum())
    ratio = cases / count * 100 if count else float("nan")
    bmi_summary_rows.append({
        "대상": group_name,
        "사람 수": count,
        "뇌졸중을 겪은 사람 수": cases,
        "뇌졸중 비율(%)": round(ratio, 2) if count else None,
    })

bmi_summary = pd.DataFrame(bmi_summary_rows)
st.dataframe(bmi_summary, use_container_width=True, hide_index=True)
st.caption("전체 비율은 stroke가 0 또는 1로 기록된 사람을 기준으로 계산했습니다.")

# --------------------------------------------------
# 5. 흡연 상태별 사람 수
# --------------------------------------------------
st.header("5. 흡연 상태별 사람 수")
smoking = data["smoking_status"].astype("string").fillna("(빈 값)")
smoking_counts = (
    smoking.value_counts(dropna=False)
    .rename_axis("흡연 상태")
    .reset_index(name="사람 수")
)
st.dataframe(smoking_counts, use_container_width=True, hide_index=True)

st.caption("이 페이지의 그래프는 데이터의 분포와 관찰된 차이를 탐색하기 위한 것입니다. 의료적 진단이나 개인의 위험 예측에 사용하지 마세요.")
'''

out = Path("/mnt/data")
pages_dir = out / "pages"
pages_dir.mkdir(exist_ok=True)
page_path = pages_dir / "1_탐색.py"
page_path.write_text(page_code, encoding="utf-8")

# Bundle the full project layout, preserving the existing main.py and requirements.txt.
zip_path = out / "뇌졸중_예측_실습실_탐색페이지_추가.zip"
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    zf.write(out / "main.py", arcname="main.py")
    zf.write(out / "requirements.txt", arcname="requirements.txt")
    zf.write(page_path, arcname="pages/1_탐색.py")

print(f"새 페이지: {page_path}")
print(f"프로젝트 묶음: {zip_path}")
print(f"페이지 코드 줄 수: {len(page_code.splitlines())}")
