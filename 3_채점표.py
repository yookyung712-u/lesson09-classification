import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

st.title("🧮 채점표")
st.caption("테스트 데이터로 두 분류 모델의 답을 채점하고, 놓친 사례와 헛짚은 사례를 살펴봅니다.")

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"
TEST_SIZE = 0.2
RANDOM_STATE = 42


@st.cache_data(ttl=3600)
def load_data():
    """첫 화면과 같은 주소에서 UTF-8로 데이터를 읽습니다."""
    return pd.read_csv(DATA_URL, encoding="utf-8")


try:
    raw_data = load_data()
except Exception as exc:
    st.error("데이터를 불러오지 못했습니다. 인터넷 연결과 데이터 주소를 확인해 주세요.")
    st.code(str(exc))
    st.stop()

needed = [
    "id", "gender", "age", "hypertension", "heart_disease",
    "ever_married", "work_type", "Residence_type",
    "avg_glucose_level", "bmi", "smoking_status", "stroke",
]
missing = [column for column in needed if column not in raw_data.columns]
if missing:
    st.error("분석에 필요한 열이 없습니다: " + ", ".join(missing))
    st.stop()

data = raw_data[needed].copy()
data["stroke"] = pd.to_numeric(data["stroke"], errors="coerce")
data = data[data["stroke"].isin([0, 1])].copy()
data["stroke"] = data["stroke"].astype(int)

if data["stroke"].nunique() != 2 or data["stroke"].value_counts().min() < 2:
    st.error("훈련용·테스트용으로 나눌 만큼 각 클래스의 데이터가 충분하지 않습니다.")
    st.stop()

# id는 식별 번호이므로 모델 입력에서 제외합니다.
# 수치 결측값은 중앙값, 범주형 결측값은 최빈값으로 채웁니다.
# 범주형은 원-핫 인코딩, 수치형은 표준화합니다.
numeric_features = [
    "age", "hypertension", "heart_disease", "avg_glucose_level", "bmi"
]
categorical_features = [
    "gender", "ever_married", "work_type", "Residence_type", "smoking_status"
]
feature_columns = numeric_features + categorical_features
X = data[feature_columns].copy()
y = data["stroke"].copy()

numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])
preprocessor = ColumnTransformer(transformers=[
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
])

try:
    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X, y, data.index, test_size=TEST_SIZE, random_state=RANDOM_STATE,
        stratify=y,
    )
    logistic_model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=2000)),
    ])
    tree_model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(random_state=RANDOM_STATE)),
    ])
    with st.spinner("두 모델을 학습하고 테스트 데이터를 채점하는 중입니다..."):
        logistic_model.fit(X_train, y_train)
        tree_model.fit(X_train, y_train)
        logistic_predictions = logistic_model.predict(X_test)
        tree_predictions = tree_model.predict(X_test)
except Exception as exc:
    st.error("데이터 분리 또는 모델 학습 중 문제가 발생했습니다.")
    st.code(str(exc))
    st.stop()

model_predictions = {
    "로지스틱 회귀(확률로 답하는 모델)": logistic_predictions,
    "의사결정트리(질문으로 답하는 모델)": tree_predictions,
}

test_n = len(y_test)
test_strokes = int((y_test == 1).sum())
st.markdown(f"**테스트용 사람 수: {test_n:,}명 · 그중 실제 뇌졸중인 사람: {test_strokes:,}명**")
st.caption(
    f"훈련용 {len(y_train):,}명, 테스트용 {test_n:,}명 "
    f"(테스트 비율 {TEST_SIZE:.0%}, 난수 시드 {RANDOM_STATE}, 층화 분리)"
)
st.info(
    "id는 모델 입력에서 제외합니다. bmi 등 수치 결측값은 훈련 자료의 중앙값, "
    "범주형 결측값은 훈련 자료의 최빈값으로 채우며 범주형 값은 원-핫 인코딩합니다."
)


def score_metrics(actual, predicted):
    tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[0, 1]).ravel()
    total = tn + fp + fn + tp
    accuracy = (tp + tn) / total if total else None
    recall = tp / (tp + fn) if (tp + fn) else None
    predicted_positive = tp + fp
    precision = tp / predicted_positive if predicted_positive else None
    if precision is None or recall is None:
        f1 = None
    elif precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "정확도": accuracy, "재현율": recall, "정밀도": precision, "F1": f1,
    }


def show_metric(value):
    return "계산할 수 없음" if value is None else f"{value:.4f}"


scores = {
    name: score_metrics(y_test, prediction)
    for name, prediction in model_predictions.items()
}

# 기준 모델은 훈련 데이터에서 더 많은 쪽 클래스만 예측합니다.
train_counts = y_train.value_counts()
majority_class = int(train_counts.idxmax())
baseline_score = score_metrics(y_test, [majority_class] * len(y_test))
scores["훈련용에서 많은 쪽으로만 답하는 모델"] = baseline_score

st.header("1. 두 모델의 혼동행렬")
st.caption("행은 실제 정답, 열은 예측입니다. 각 칸에 사람 수와 TP·FP·FN·TN을 함께 표시합니다.")


def confusion_display(score):
    tn, fp, fn, tp = score["TN"], score["FP"], score["FN"], score["TP"]
    return pd.DataFrame(
        [
            [f"TN: {tn:,}명", f"FP: {fp:,}명", f"{tn + fp:,}명"],
            [f"FN: {fn:,}명", f"TP: {tp:,}명", f"{fn + tp:,}명"],
            [f"{tn + fn:,}명", f"{fp + tp:,}명", f"{tn + fp + fn + tp:,}명"],
        ],
        index=["실제 뇌졸중 아님(0)", "실제 뇌졸중(1)", "열 합계"],
        columns=["예측 뇌졸중 아님(0)", "예측 뇌졸중(1)", "행 합계"],
    )


left_matrix, right_matrix = st.columns(2)
with left_matrix:
    st.subheader("로지스틱 회귀(확률로 답하는 모델)")
    st.dataframe(confusion_display(scores["로지스틱 회귀(확률로 답하는 모델)"]), use_container_width=True)
with right_matrix:
    st.subheader("의사결정트리(질문으로 답하는 모델)")
    st.dataframe(confusion_display(scores["의사결정트리(질문으로 답하는 모델)"]), use_container_width=True)

st.caption("행 합계는 실제 그룹, 열 합계는 해당 예측을 받은 사람 수입니다. 오른쪽 아래는 테스트 전체입니다.")

st.header("2. 네 가지 지표 비교")
metric_rows = []
for name, score in scores.items():
    metric_rows.append({
        "모델": name,
        "정확도": show_metric(score["정확도"]),
        "재현율": show_metric(score["재현율"]),
        "정밀도": show_metric(score["정밀도"]),
        "F1": show_metric(score["F1"]),
    })
st.dataframe(pd.DataFrame(metric_rows), use_container_width=True, hide_index=True)
st.caption(
    "정확도=(TP+TN)/전체, 재현율=TP/(TP+FN), 정밀도=TP/(TP+FP), "
    "F1=2×정밀도×재현율/(정밀도+재현율). 뇌졸중 예측이 0명이면 "
    "정밀도와 F1은 ‘계산할 수 없음’으로 표시합니다."
)

st.header("3. 놓친 사람과 헛짚은 사람")
chosen_model = st.selectbox("사례 목록을 볼 모델을 선택하세요.", list(model_predictions))
test_cases = data.loc[test_idx].copy()
test_cases["실제"] = y_test.to_numpy()
test_cases["예측"] = model_predictions[chosen_model]
test_cases["번호"] = test_cases["id"]

missed = test_cases[(test_cases["실제"] == 1) & (test_cases["예측"] == 0)].copy()
false_alarms = test_cases[(test_cases["실제"] == 0) & (test_cases["예측"] == 1)].copy()
columns = ["번호", "age", "avg_glucose_level", "hypertension", "heart_disease"]
renames = {
    "번호": "번호", "age": "나이", "avg_glucose_level": "평균 혈당",
    "hypertension": "고혈압 여부", "heart_disease": "심장병 여부",
}


def format_cases(frame):
    result = frame.sort_values("age", ascending=False, na_position="last")[columns].rename(columns=renames).copy()
    result["나이"] = result["나이"].round(1)
    result["평균 혈당"] = result["평균 혈당"].round(1)
    for col in ["고혈압 여부", "심장병 여부"]:
        result[col] = result[col].map({0: "없음(0)", 1: "있음(1)"}).fillna("알 수 없음")
    return result


col_missed, col_fp = st.columns(2)
with col_missed:
    st.subheader(f"놓친 환자 · FN {len(missed):,}명")
    st.caption("실제 뇌졸중인데 아니라고 예측한 사람")
    missed_table = format_cases(missed)
    st.dataframe(missed_table.head(20), use_container_width=True, hide_index=True)
    if len(missed_table) > 20:
        st.caption(f"전체 {len(missed_table):,}명 중 나이가 많은 순으로 20명만 표시했습니다.")
    elif len(missed_table) == 0:
        st.caption("해당 사례가 없습니다.")

with col_fp:
    st.subheader(f"헛짚은 사람 · FP {len(false_alarms):,}명")
    st.caption("실제 뇌졸중이 아닌데 뇌졸중이라고 예측한 사람")
    fp_table = format_cases(false_alarms)
    st.dataframe(fp_table.head(20), use_container_width=True, hide_index=True)
    if len(fp_table) > 20:
        st.caption(f"전체 {len(fp_table):,}명 중 나이가 많은 순으로 20명만 표시했습니다.")
    elif len(fp_table) == 0:
        st.caption("해당 사례가 없습니다.")

st.markdown(
    f"**목록 확인: 놓친 환자 {len(missed):,}명 = 혼동행렬 FN {scores[chosen_model]['FN']:,}명, "
    f"헛짚은 사람 {len(false_alarms):,}명 = 혼동행렬 FP {scores[chosen_model]['FP']:,}명입니다.**"
)
st.warning("이 결과는 교육용입니다. 의료 진단이나 개인의 실제 건강 위험 판단에 사용하지 마세요.")
