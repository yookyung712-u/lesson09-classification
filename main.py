from pathlib import Path
import json

import pandas as pd
import plotly.express as px  # 실습에서 그래프를 추가할 때 사용할 수 있습니다.
import streamlit as st
from sklearn.model_selection import train_test_split  # 다음 실습에서 활용할 수 있습니다.

# -----------------------------
# 기본 설정
# -----------------------------
st.set_page_config(
    page_title="뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 뇌졸중 예측 실습실")
st.caption("데이터를 살펴보고, 열의 의미를 직접 정리해 보는 첫걸음")

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"
RECORD_DIR = Path("내_기록")
RECORD_FILE = RECORD_DIR / "stroke_데이터_소개.json"

COLUMNS = [
    "id", "gender", "age", "hypertension", "heart_disease",
    "ever_married", "work_type", "Residence_type",
    "avg_glucose_level", "bmi", "smoking_status", "stroke",
]


@st.cache_data(ttl=3600)
def load_data():
    """원본 CSV를 UTF-8로 읽습니다."""
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    # 교재에 안내된 열 구성을 기준으로 하되, 데이터에 없는 열은 강제로 만들지 않습니다.
    return df


def load_notes():
    """사용자가 저장한 열 뜻과 출처를 불러옵니다."""
    if not RECORD_FILE.exists():
        return {}, ""
    try:
        with RECORD_FILE.open("r", encoding="utf-8") as file:
            saved = json.load(file)
        return saved.get("meanings", {}), saved.get("source", "")
    except (OSError, json.JSONDecodeError, AttributeError):
        st.warning("저장된 기록 파일을 읽지 못해 빈 기록으로 시작합니다.")
        return {}, ""


try:
    data = load_data()
except Exception as exc:
    st.error("데이터를 불러오지 못했습니다. 인터넷 연결과 데이터 주소를 확인해 주세요.")
    st.code(str(exc))
    st.stop()

missing_columns = [name for name in COLUMNS if name not in data.columns]
if missing_columns:
    st.warning(f"예상한 열 중 데이터에 없는 열이 있습니다: {', '.join(missing_columns)}")

total_people = len(data)
column_count = len(data.columns)
stroke_count = int((pd.to_numeric(data["stroke"], errors="coerce") == 1).sum()) if "stroke" in data else 0
stroke_ratio = (stroke_count / total_people * 100) if total_people else 0.0

# -----------------------------
# 숫자 요약 카드
# -----------------------------
st.subheader("데이터 한눈에 보기")
metric_cols = st.columns(4)
metric_cols[0].metric("전체 사람 수", f"{total_people:,}명")
metric_cols[1].metric("열 개수", f"{column_count:,}개")
metric_cols[2].metric("stroke가 1인 사람 수", f"{stroke_count:,}명")
metric_cols[3].metric("stroke가 1인 비율", f"{stroke_ratio:.2f}%")

st.info(
    "이 화면은 학습용 데이터 소개입니다. 데이터의 예측 결과는 실제 개인의 뇌졸중 위험을 "
    "진단하거나 의료적 판단을 대신하지 않습니다."
)

# -----------------------------
# 열 설명표: 뜻 칸은 직접 입력
# -----------------------------
st.subheader("열 이름과 값 살펴보기")
st.write("‘우리말 뜻’ 칸을 직접 채워 보세요. 저장 버튼을 누르면 기록 파일에 남습니다.")

saved_meanings, saved_source = load_notes()
rows = []
for col in data.columns:
    rows.append({
        "열 이름": str(col),
        "우리말 뜻": str(saved_meanings.get(str(col), "")),
        "값의 종류": int(data[col].nunique(dropna=True)),
        "빈 값 개수": int(data[col].isna().sum()),
    })

description = pd.DataFrame(rows, columns=["열 이름", "우리말 뜻", "값의 종류", "빈 값 개수"])
edited_description = st.data_editor(
    description,
    hide_index=True,
    use_container_width=True,
    disabled=["열 이름", "값의 종류", "빈 값 개수"],
    column_config={
        "열 이름": st.column_config.TextColumn("열 이름"),
        "우리말 뜻": st.column_config.TextColumn("우리말 뜻", help="교재를 보고 뜻을 입력하세요."),
        "값의 종류": st.column_config.NumberColumn("값의 종류", format="%d"),
        "빈 값 개수": st.column_config.NumberColumn("빈 값 개수", format="%d"),
    },
    key="column_description_editor",
)

st.subheader("데이터 처음 다섯 줄")
st.dataframe(data.head(5), use_container_width=True, hide_index=True)

# -----------------------------
# 출처 기록 및 저장
# -----------------------------
st.subheader("데이터 출처")
source_text = st.text_area(
    "교재에 나온 데이터 출처를 적어 주세요.",
    value=saved_source,
    placeholder="교재의 출처 안내를 여기에 입력하세요.",
    height=100,
)

if st.button("우리말 뜻과 출처 저장", type="primary"):
    meanings = {
        str(row["열 이름"]): str(row["우리말 뜻"]).strip()
        for _, row in edited_description.iterrows()
    }
    payload = {
        "meanings": meanings,
        "source": source_text.strip(),
    }
    try:
        RECORD_DIR.mkdir(parents=True, exist_ok=True)
        with RECORD_FILE.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
        st.success(f"기록을 저장했습니다: {RECORD_FILE.as_posix()}")
    except OSError as exc:
        st.error("기록 파일을 저장하지 못했습니다. 실행 환경의 쓰기 권한을 확인해 주세요.")
        st.code(str(exc))

st.caption(
    "기록은 앱이 실행 중인 환경의 내_기록/stroke_데이터_소개.json 파일에 저장됩니다. "
    "새로고침하면 이 파일에서 다시 불러옵니다. 다만 Streamlit Cloud에서는 재배포·재시작 시 "
    "로컬 파일이 유지되지 않을 수 있으니, 중요한 기록은 별도로 백업하세요."
)
