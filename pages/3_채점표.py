# pages/3_채점표.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix


# --------------------------------------------------
# 페이지 설정
# --------------------------------------------------
st.set_page_config(
    page_title="채점표",
    page_icon="🧮",
    layout="wide"
)

st.title("🧮 채점표")
st.write(
    "테스트 데이터에서 두 분류 모델의 예측을 실제 정답과 비교합니다. "
    "혼동행렬의 행은 실제 결과, 열은 모델의 예측 결과입니다."
)


# --------------------------------------------------
# 데이터 불러오기
# --------------------------------------------------
DATA_URL = "https://raw.githubusercontent.com/s200ua1/data/refs/heads/main/stroke.csv"


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    # id 순서대로 정렬
    df = df.sort_values("id").reset_index(drop=True)

    return df


try:
    df = load_data()
except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()


# --------------------------------------------------
# 사용할 특성
# --------------------------------------------------
FEATURES = [
    "age",
    "avg_glucose_level",
    "hypertension",
    "heart_disease"
]

FEATURE_NAMES = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "hypertension": "고혈압 여부",
    "heart_disease": "심장병 여부"
}

TARGET = "stroke"


# --------------------------------------------------
# 데이터 나누기
# id 기준 10명씩 묶고, 각 묶음의 처음 3명을 테스트용으로 사용
# --------------------------------------------------
test_mask = np.zeros(len(df), dtype=bool)

for start in range(0, len(df), 10):
    end = min(start + 10, len(df))
    test_mask[start:min(start + 3, end)] = True

train_df = df.loc[~test_mask].copy()
test_df = df.loc[test_mask].copy()

X_train = train_df[FEATURES].copy()
X_test = test_df[FEATURES].copy()

y_train = train_df[TARGET].astype(int).copy()
y_test = test_df[TARGET].astype(int).copy()


# --------------------------------------------------
# 전처리
# 결측치 대체와 표준화는 훈련용 데이터에 맞춰 학습
# --------------------------------------------------
imputer = SimpleImputer(strategy="median")

X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)


# --------------------------------------------------
# 모델 학습
# --------------------------------------------------
logistic_model = LogisticRegression(
    random_state=42,
    max_iter=1000
)

tree_model = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    random_state=42
)

logistic_model.fit(X_train_scaled, y_train)
tree_model.fit(X_train_imputed, y_train)


# --------------------------------------------------
# 예측
# --------------------------------------------------
logistic_pred = logistic_model.predict(X_test_scaled)
tree_pred = tree_model.predict(X_test_imputed)


# --------------------------------------------------
# 훈련용에서 많은 쪽으로만 답하는 기준 모델
# --------------------------------------------------
train_majority_class = int(y_train.value_counts().idxmax())

baseline_pred = np.full(
    len(y_test),
    train_majority_class,
    dtype=int
)


# --------------------------------------------------
# 혼동행렬과 지표 계산 함수
# --------------------------------------------------
def calculate_metrics(y_true, y_pred):
    """
    혼동행렬:
        TN: 실제 0, 예측 0
        FP: 실제 0, 예측 1
        FN: 실제 1, 예측 0
        TP: 실제 1, 예측 1
    """

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    total = tn + fp + fn + tp

    accuracy = (tp + tn) / total if total > 0 else np.nan

    actual_positive = tp + fn
    predicted_positive = tp + fp

    recall = (
        tp / actual_positive
        if actual_positive > 0
        else np.nan
    )

    if predicted_positive == 0:
        precision = None
        f1 = None
    else:
        precision = tp / predicted_positive

        if recall is None or np.isnan(recall):
            f1 = None
        elif precision + recall == 0:
            f1 = 0.0
        else:
            f1 = (
                2 * precision * recall
                / (precision + recall)
            )

    return {
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "정확도": accuracy,
        "재현율": recall,
        "정밀도": precision,
        "F1": f1
    }


logistic_metrics = calculate_metrics(y_test, logistic_pred)
tree_metrics = calculate_metrics(y_test, tree_pred)
baseline_metrics = calculate_metrics(y_test, baseline_pred)


# --------------------------------------------------
# 화면 상단: 테스트 데이터 요약
# --------------------------------------------------
test_count = len(test_df)
test_stroke_count = int(y_test.sum())

st.markdown("### 테스트 데이터")

st.write(
    f"테스트용 사람 수: **{test_count:,}명** "
    f"｜ 그중 실제 뇌졸중인 사람: **{test_stroke_count:,}명**"
)


# --------------------------------------------------
# 혼동행렬 표시 함수
# 행: 실제, 열: 예측
# --------------------------------------------------
def show_confusion_matrix(title, metrics):
    st.markdown(f"#### {title}")

    matrix_df = pd.DataFrame(
        [
            [
                f"TN: {metrics['TN']:,}명",
                f"FP: {metrics['FP']:,}명",
                f"합계: {metrics['TN'] + metrics['FP']:,}명"
            ],
            [
                f"FN: {metrics['FN']:,}명",
                f"TP: {metrics['TP']:,}명",
                f"합계: {metrics['FN'] + metrics['TP']:,}명"
            ],
            [
                f"합계: {metrics['TN'] + metrics['FN']:,}명",
                f"합계: {metrics['FP'] + metrics['TP']:,}명",
                f"전체: {metrics['TN'] + metrics['FP'] + metrics['FN'] + metrics['TP']:,}명"
            ]
        ],
        index=[
            "실제: 뇌졸중 아님 (0)",
            "실제: 뇌졸중 (1)",
            "합계"
        ],
        columns=[
            "예측: 뇌졸중 아님 (0)",
            "예측: 뇌졸중 (1)",
            "행 합계"
        ]
    )

    st.table(matrix_df)


# --------------------------------------------------
# 두 모델 혼동행렬
# --------------------------------------------------
st.markdown("### 혼동행렬")

col1, col2 = st.columns(2)

with col1:
    show_confusion_matrix(
        "로지스틱 회귀(확률로 답하는 모델)",
        logistic_metrics
    )

with col2:
    show_confusion_matrix(
        "의사결정트리(질문으로 답하는 모델)",
        tree_metrics
    )


# --------------------------------------------------
# 지표 표
# --------------------------------------------------
def format_metric(value):
    if value is None:
        return "계산할 수 없음"

    if isinstance(value, float) and np.isnan(value):
        return "계산할 수 없음"

    return f"{value:.3f}"


metrics_table = pd.DataFrame(
    [
        {
            "모델": "로지스틱 회귀(확률로 답하는 모델)",
            "정확도": format_metric(logistic_metrics["정확도"]),
            "재현율": format_metric(logistic_metrics["재현율"]),
            "정밀도": format_metric(logistic_metrics["정밀도"]),
            "F1": format_metric(logistic_metrics["F1"])
        },
        {
            "모델": "의사결정트리(질문으로 답하는 모델)",
            "정확도": format_metric(tree_metrics["정확도"]),
            "재현율": format_metric(tree_metrics["재현율"]),
            "정밀도": format_metric(tree_metrics["정밀도"]),
            "F1": format_metric(tree_metrics["F1"])
        },
        {
            "모델": "훈련용에서 많은 쪽으로만 답하는 모델",
            "정확도": format_metric(baseline_metrics["정확도"]),
            "재현율": format_metric(baseline_metrics["재현율"]),
            "정밀도": format_metric(baseline_metrics["정밀도"]),
            "F1": format_metric(baseline_metrics["F1"])
        }
    ]
)

st.markdown("### 성능 지표 비교")
st.dataframe(
    metrics_table,
    use_container_width=True,
    hide_index=True
)

st.caption(
    "정확도 = (TP + TN) / 전체 사람 수 · "
    "재현율 = TP / 실제 뇌졸중인 사람 수 · "
    "정밀도 = TP / 뇌졸중이라 예측한 사람 수 · "
    "F1 = 2 × 정밀도 × 재현율 / (정밀도 + 재현율)"
)

st.caption(
    "뇌졸중이라 예측한 사람이 0명이면 정밀도와 F1은 "
    "'계산할 수 없음'으로 표시합니다."
)


# --------------------------------------------------
# 놓친 환자 / 헛짚은 사람 목록 생성
# --------------------------------------------------
def make_error_list(predictions, error_type):
    result_df = test_df.copy()
    result_df["실제"] = y_test.values
    result_df["예측"] = predictions

    if error_type == "FN":
        # 실제 뇌졸중(1), 예측은 뇌졸중 아님(0)
        result_df = result_df[
            (result_df["실제"] == 1)
            & (result_df["예측"] == 0)
        ].copy()

    elif error_type == "FP":
        # 실제 뇌졸중 아님(0), 예측은 뇌졸중(1)
        result_df = result_df[
            (result_df["실제"] == 0)
            & (result_df["예측"] == 1)
        ].copy()

    result_df = result_df.sort_values(
        "age",
        ascending=False
    )

    result_df = result_df.rename(
        columns={
            "id": "번호",
            "age": "나이",
            "avg_glucose_level": "평균 혈당",
            "hypertension": "고혈압 여부",
            "heart_disease": "심장병 여부"
        }
    )

    result_df["고혈압 여부"] = result_df["고혈압 여부"].map(
        {0: "아니오", 1: "예"}
    )

    result_df["심장병 여부"] = result_df["심장병 여부"].map(
        {0: "아니오", 1: "예"}
    )

    columns = [
        "번호",
        "나이",
        "평균 혈당",
        "고혈압 여부",
        "심장병 여부"
    ]

    return result_df[columns]


# --------------------------------------------------
# 모델 선택
# --------------------------------------------------
st.markdown("### 놓친 환자와 헛짚은 사람")

selected_model = st.selectbox(
    "목록을 확인할 모델을 선택하세요.",
    [
        "로지스틱 회귀(확률로 답하는 모델)",
        "의사결정트리(질문으로 답하는 모델)"
    ]
)

if selected_model == "로지스틱 회귀(확률로 답하는 모델)":
    selected_predictions = logistic_pred
    selected_metrics = logistic_metrics
else:
    selected_predictions = tree_pred
    selected_metrics = tree_metrics


fn_list = make_error_list(
    selected_predictions,
    "FN"
)

fp_list = make_error_list(
    selected_predictions,
    "FP"
)


# --------------------------------------------------
# 오류 목록 나란히 표시
# --------------------------------------------------
left_col, right_col = st.columns(2)

with left_col:
    st.markdown(
        f"#### 놓친 환자 ({len(fn_list):,}명)"
    )
    st.write(
        "실제로 뇌졸중이지만, 모델이 뇌졸중이 아니라고 예측한 사람입니다."
    )

    if len(fn_list) == 0:
        st.info("놓친 환자가 없습니다.")
    else:
        st.dataframe(
            fn_list.head(20),
            use_container_width=True,
            hide_index=True
        )

        if len(fn_list) > 20:
            st.caption(
                f"전체 {len(fn_list):,}명 중 나이가 많은 순서로 "
                "앞의 20명만 표시했습니다."
            )

with right_col:
    st.markdown(
        f"#### 헛짚은 사람 ({len(fp_list):,}명)"
    )
    st.write(
        "실제로 뇌졸중이 아니지만, 모델이 뇌졸중이라고 예측한 사람입니다."
    )

    if len(fp_list) == 0:
        st.info("헛짚은 사람이 없습니다.")
    else:
        st.dataframe(
            fp_list.head(20),
            use_container_width=True,
            hide_index=True
        )

        if len(fp_list) > 20:
            st.caption(
                f"전체 {len(fp_list):,}명 중 나이가 많은 순서로 "
                "앞의 20명만 표시했습니다."
            )


st.caption(
    f"목록의 인원수는 혼동행렬의 FN(놓친 환자) "
    f"{selected_metrics['FN']:,}명 및 FP(헛짚은 사람) "
    f"{selected_metrics['FP']:,}명과 각각 같습니다."
)


# --------------------------------------------------
# Plotly 시각화: 두 모델 지표 비교
# --------------------------------------------------
st.markdown("### 두 모델 지표 그래프")

plot_metrics = pd.DataFrame(
    {
        "지표": ["정확도", "재현율", "정밀도", "F1"],
        "로지스틱 회귀": [
            logistic_metrics["정확도"],
            logistic_metrics["재현율"],
            logistic_metrics["정밀도"],
            logistic_metrics["F1"]
        ],
        "의사결정트리": [
            tree_metrics["정확도"],
            tree_metrics["재현율"],
            tree_metrics["정밀도"],
            tree_metrics["F1"]
        ]
    }
)

fig = go.Figure()

fig.add_trace(
    go.Bar(
        x=plot_metrics["지표"],
        y=plot_metrics["로지스틱 회귀"],
        name="로지스틱 회귀(확률로 답하는 모델)"
    )
)

fig.add_trace(
    go.Bar(
        x=plot_metrics["지표"],
        y=plot_metrics["의사결정트리"],
        name="의사결정트리(질문으로 답하는 모델)"
    )
)

fig.update_layout(
    barmode="group",
    yaxis_title="점수",
    yaxis_range=[0, 1],
    xaxis_title="평가 지표",
    legend_title="모델",
    height=450
)

st.plotly_chart(
    fig,
    use_container_width=True
)
