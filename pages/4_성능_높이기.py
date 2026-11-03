# pages/4_성능_높이기.py

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
    page_title="성능 높이기",
    page_icon="📈",
    layout="wide"
)

st.title("📈 성능 높이기")
st.write(
    "뇌졸중을 겪은 사람이 드문 데이터에서 "
    "두 모델의 가중치 조정 전·후 성능을 비교합니다."
)


# --------------------------------------------------
# 데이터 불러오기
# --------------------------------------------------
DATA_URL = (
    "https://raw.githubusercontent.com/"
    "s200ua1/data/refs/heads/main/stroke.csv"
)


@st.cache_data
def load_data():
    data = pd.read_csv(DATA_URL, encoding="utf-8")
    data = data.sort_values("id").reset_index(drop=True)
    return data


try:
    df = load_data()
except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()


# --------------------------------------------------
# 입력 속성 선택
# --------------------------------------------------
FEATURE_NAMES = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수(BMI)",
    "hypertension": "고혈압 여부",
    "heart_disease": "심장병 여부"
}

FEATURE_OPTIONS = list(FEATURE_NAMES.keys())

st.markdown("### 입력으로 사용할 속성 선택")

selected_features = st.multiselect(
    "모델에 넣을 속성을 선택하세요.",
    options=FEATURE_OPTIONS,
    default=[
        "age",
        "avg_glucose_level",
        "hypertension",
        "heart_disease"
    ],
    format_func=lambda feature: FEATURE_NAMES[feature]
)

if len(selected_features) < 2:
    st.warning("속성을 두 개 이상 선택해 주세요.")
    st.stop()


# --------------------------------------------------
# 데이터 분할
# 채점표 페이지와 동일하게 id 순서로 10명씩 묶고
# 각 묶음의 처음 3명을 테스트용으로 사용
# --------------------------------------------------
test_mask = np.zeros(len(df), dtype=bool)

for start in range(0, len(df), 10):
    end = min(start + 10, len(df))
    test_mask[start:min(start + 3, end)] = True

train_df = df.loc[~test_mask].copy()
test_df = df.loc[test_mask].copy()

X_train = train_df[selected_features].copy()
X_test = test_df[selected_features].copy()

y_train = train_df["stroke"].astype(int).copy()
y_test = test_df["stroke"].astype(int).copy()


# --------------------------------------------------
# 결측치 대체
# 선택한 속성에 BMI가 포함되어 있을 경우
# 훈련용 데이터의 중앙값으로 결측치 대체
# --------------------------------------------------
imputer = SimpleImputer(strategy="median")

X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)


# --------------------------------------------------
# 표준화
# 표준화 기준도 훈련용 데이터로만 학습
# --------------------------------------------------
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)


# --------------------------------------------------
# 모델 학습
# 가중치 조정 전: class_weight=None
# 가중치 조정 후: class_weight="balanced"
# --------------------------------------------------
def train_models(class_weight):
    logistic = LogisticRegression(
        class_weight=class_weight,
        random_state=42,
        max_iter=1000
    )

    tree = DecisionTreeClassifier(
        class_weight=class_weight,
        max_depth=3,
        min_samples_leaf=5,
        random_state=42
    )

    logistic.fit(X_train_scaled, y_train)
    tree.fit(X_train_imputed, y_train)

    return logistic, tree


logistic_before, tree_before = train_models(None)
logistic_after, tree_after = train_models("balanced")


# --------------------------------------------------
# 예측
# --------------------------------------------------
predictions = {
    "logistic_before": logistic_before.predict(X_test_scaled),
    "logistic_after": logistic_after.predict(X_test_scaled),
    "tree_before": tree_before.predict(X_test_imputed),
    "tree_after": tree_after.predict(X_test_imputed)
}


# --------------------------------------------------
# 혼동행렬과 지표 계산
# --------------------------------------------------
def calculate_metrics(y_true, y_pred):
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    total = tn + fp + fn + tp

    accuracy = (
        (tp + tn) / total
        if total > 0
        else np.nan
    )

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

        if np.isnan(recall):
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


metrics = {
    name: calculate_metrics(y_test, pred)
    for name, pred in predictions.items()
}


# --------------------------------------------------
# 테스트 데이터 요약
# --------------------------------------------------
st.markdown("### 테스트 데이터")

st.write(
    f"테스트용 사람 수: **{len(test_df):,}명** "
    f"｜ 실제 뇌졸중인 사람 수: **{int(y_test.sum()):,}명**"
)

st.caption(
    "가중치 조정은 훈련용 데이터에서만 적용합니다. "
    "테스트용 데이터는 두 모델의 결과를 평가하는 데 사용합니다."
)


# --------------------------------------------------
# 표시용 함수
# --------------------------------------------------
def format_metric(value):
    if value is None:
        return "계산할 수 없음"

    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return "계산할 수 없음"

    return f"{value:.3f}"


def confusion_cell(label, count):
    return f"{label}: {count:,}명"


def confusion_table(metric):
    return pd.DataFrame(
        [
            [
                confusion_cell("TN", metric["TN"]),
                confusion_cell("FP", metric["FP"]),
                f"합계: {metric['TN'] + metric['FP']:,}명"
            ],
            [
                confusion_cell("FN", metric["FN"]),
                confusion_cell("TP", metric["TP"]),
                f"합계: {metric['FN'] + metric['TP']:,}명"
            ],
            [
                f"합계: {metric['TN'] + metric['FN']:,}명",
                f"합계: {metric['FP'] + metric['TP']:,}명",
                f"전체: {metric['TN'] + metric['FP'] + metric['FN'] + metric['TP']:,}명"
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


# --------------------------------------------------
# 두 모델의 혼동행렬 및 지표
# --------------------------------------------------
st.markdown("### 혼동행렬과 지표 비교")

model_specs = [
    (
        "로지스틱 회귀(확률로 답하는 모델)",
        "logistic_before",
        "logistic_after"
    ),
    (
        "의사결정트리(질문으로 답하는 모델)",
        "tree_before",
        "tree_after"
    )
]

for model_name, before_key, after_key in model_specs:
    st.markdown(f"#### {model_name}")

    before_col, after_col = st.columns(2)

    with before_col:
        st.markdown("**가중치 조정 전**")
        st.table(confusion_table(metrics[before_key]))

    with after_col:
        st.markdown("**가중치 조정 후**")
        st.table(confusion_table(metrics[after_key]))

    metric_table = pd.DataFrame(
        [
            {
                "상태": "가중치 조정 전",
                "정확도": format_metric(metrics[before_key]["정확도"]),
                "재현율": format_metric(metrics[before_key]["재현율"]),
                "정밀도": format_metric(metrics[before_key]["정밀도"]),
                "F1": format_metric(metrics[before_key]["F1"])
            },
            {
                "상태": "가중치 조정 후",
                "정확도": format_metric(metrics[after_key]["정확도"]),
                "재현율": format_metric(metrics[after_key]["재현율"]),
                "정밀도": format_metric(metrics[after_key]["정밀도"]),
                "F1": format_metric(metrics[after_key]["F1"])
            }
        ]
    )

    st.dataframe(
        metric_table,
        use_container_width=True,
        hide_index=True
    )

    st.divider()


# --------------------------------------------------
# 환자 발견·놓침·헛짚음 변화
# --------------------------------------------------
st.markdown("### 환자 발견 결과 변화")

summary_rows = []

for model_name, before_key, after_key in model_specs:
    before = metrics[before_key]
    after = metrics[after_key]

    summary_rows.append(
        {
            "모델": model_name,
            "찾아낸 환자 수(TP)": (
                f"{before['TP']:,}명 → {after['TP']:,}명"
            ),
            "놓친 환자 수(FN)": (
                f"{before['FN']:,}명 → {after['FN']:,}명"
            ),
            "헛짚은 사람 수(FP)": (
                f"{before['FP']:,}명 → {after['FP']:,}명"
            )
        }
    )

summary_df = pd.DataFrame(summary_rows)

st.dataframe(
    summary_df,
    use_container_width=True,
    hide_index=True
)

st.caption(
    "찾아낸 환자 = TP(실제 뇌졸중이고 뇌졸중이라고 예측), "
    "놓친 환자 = FN(실제 뇌졸중인데 아니라고 예측), "
    "헛짚은 사람 = FP(실제는 아니지만 뇌졸중이라고 예측)"
)


# --------------------------------------------------
# Plotly 경계 시각화 준비
# --------------------------------------------------
st.markdown("### 실제 결과와 0.5 경계 비교")

if len(selected_features) == 2:
    st.write(
        "선택한 두 속성을 가로·세로축으로 사용합니다. "
        "다른 속성은 테스트 데이터의 중앙값으로 고정합니다."
    )
else:
    st.write(
        "선택한 속성 중 처음 세 개를 3차원 축으로 사용합니다. "
        "그 외의 속성은 테스트 데이터의 중앙값으로 고정합니다. "
        "그래프를 마우스로 돌려 볼 수 있습니다."
    )


# 테스트 데이터의 실제 값은 원래 단위로 표시
feature_positions = {
    feature: selected_features.index(feature)
    for feature in selected_features
}

median_values = np.median(
    X_test_imputed,
    axis=0
)


def make_grid_values(feature, count=70):
    position = feature_positions[feature]
    values = X_test_imputed[:, position]

    low = float(np.min(values))
    high = float(np.max(values))

    if low == high:
        padding = max(abs(low) * 0.05, 1.0)
        low -= padding
        high += padding
    else:
        padding = (high - low) * 0.05
        low -= padding
        high += padding

    return np.linspace(low, high, count)


def make_model_input(grid_points):
    """
    원래 단위의 grid_points를 입력받아
    선택한 속성 순서에 맞는 입력 배열을 만듭니다.
    """
    grid_points = np.asarray(grid_points)

    if grid_points.ndim == 1:
        grid_points = grid_points.reshape(1, -1)

    model_input = np.tile(
        median_values,
        (len(grid_points), 1)
    )

    for axis_index, feature in enumerate(plot_features):
        model_input[:, feature_positions[feature]] = (
            grid_points[:, axis_index]
        )

    return model_input


def predict_probability(model, model_kind, grid_points):
    model_input = make_model_input(grid_points)

    if model_kind == "logistic":
        model_input_scaled = scaler.transform(model_input)
        return model.predict_proba(model_input_scaled)[:, 1]

    return model.predict_proba(model_input)[:, 1]


def get_2d_boundary(model, model_kind, x_values, y_values):
    xx, yy = np.meshgrid(x_values, y_values)

    grid_points = np.column_stack(
        [
            xx.ravel(),
            yy.ravel()
        ]
    )

    probabilities = predict_probability(
        model,
        model_kind,
        grid_points
    )

    return xx, yy, probabilities.reshape(xx.shape)


def get_3d_boundary(model, model_kind, x_values, y_values, z_values):
    xx, yy, zz = np.meshgrid(
        x_values,
        y_values,
        z_values,
        indexing="ij"
    )

    grid_points = np.column_stack(
        [
            xx.ravel(),
            yy.ravel(),
            zz.ravel()
        ]
    )

    probabilities = predict_probability(
        model,
        model_kind,
        grid_points
    )

    return xx, yy, zz, probabilities.reshape(xx.shape)


# --------------------------------------------------
# 2차원 그래프
# --------------------------------------------------
if len(selected_features) == 2:
    plot_features = selected_features[:2]

    x_feature = plot_features[0]
    y_feature = plot_features[1]

    x_values = make_grid_values(x_feature)
    y_values = make_grid_values(y_feature)

    fig = go.Figure()

    # 실제 뇌졸중이 아닌 사람
    no_stroke = y_test.to_numpy() == 0

    fig.add_trace(
        go.Scatter(
            x=X_test_imputed[no_stroke, feature_positions[x_feature]],
            y=X_test_imputed[no_stroke, feature_positions[y_feature]],
            mode="markers",
            name="실제 뇌졸중 아님",
            marker=dict(
                color="#4C78A8",
                size=8,
                opacity=0.7
            )
        )
    )

    # 실제 뇌졸중인 사람
    stroke = y_test.to_numpy() == 1

    fig.add_trace(
        go.Scatter(
            x=X_test_imputed[stroke, feature_positions[x_feature]],
            y=X_test_imputed[stroke, feature_positions[y_feature]],
            mode="markers",
            name="실제 뇌졸중",
            marker=dict(
                color="#E45756",
                size=9,
                symbol="diamond",
                opacity=0.8
            )
        )
    )

    boundary_configs = [
        (
            "로지스틱 회귀(확률로 답하는 모델)",
            logistic_before,
            "logistic",
            "가중치 조정 전",
            "#F58518"
        ),
        (
            "로지스틱 회귀(확률로 답하는 모델)",
            logistic_after,
            "logistic",
            "가중치 조정 후",
            "#B279A2"
        ),
        (
            "의사결정트리(질문으로 답하는 모델)",
            tree_before,
            "tree",
            "가중치 조정 전",
            "#54A24B"
        ),
        (
            "의사결정트리(질문으로 답하는 모델)",
            tree_after,
            "tree",
            "가중치 조정 후",
            "#72B7B2"
        )
    ]

    boundary_notes = []

    for model_name, model, model_kind, state, color in boundary_configs:
        xx, yy, probability_grid = get_2d_boundary(
            model,
            model_kind,
            x_values,
            y_values
        )

        min_probability = float(np.nanmin(probability_grid))
        max_probability = float(np.nanmax(probability_grid))

        if min_probability <= 0.5 <= max_probability:
            fig.add_trace(
                go.Contour(
                    x=x_values,
                    y=y_values,
                    z=probability_grid,
                    contours=dict(
                        start=0.5,
                        end=0.5,
                        size=1,
                        coloring="lines"
                    ),
                    line=dict(
                        color=color,
                        width=3,
                        dash="solid" if state == "가중치 조정 전" else "dash"
                    ),
                    showscale=False,
                    name=f"{model_name} - {state} 0.5 경계",
                    hoverinfo="skip"
                )
            )
        else:
            boundary_notes.append(
                f"{model_name} {state}: "
                "0.5 경계선이 현재 그림 범위 밖에 있습니다."
            )

    fig.update_layout(
        xaxis_title=FEATURE_NAMES[x_feature],
        yaxis_title=FEATURE_NAMES[y_feature],
        height=650,
        legend_title="실제 결과 및 경계선",
        title="테스트 데이터와 0.5 경계선"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    for note in boundary_notes:
        st.warning(note)

    st.caption(
        "실선은 가중치 조정 전, 점선은 가중치 조정 후의 0.5 경계선입니다. "
        "경계선은 뇌졸중일 확률이 0.5인 위치를 나타냅니다."
    )


# --------------------------------------------------
# 3차원 그래프
# --------------------------------------------------
else:
    plot_features = selected_features[:3]

    x_feature = plot_features[0]
    y_feature = plot_features[1]
    z_feature = plot_features[2]

    x_values = make_grid_values(x_feature, count=24)
    y_values = make_grid_values(y_feature, count=24)
    z_values = make_grid_values(z_feature, count=24)

    fig = go.Figure()

    no_stroke = y_test.to_numpy() == 0
    stroke = y_test.to_numpy() == 1

    fig.add_trace(
        go.Scatter3d(
            x=X_test_imputed[
                no_stroke,
                feature_positions[x_feature]
            ],
            y=X_test_imputed[
                no_stroke,
                feature_positions[y_feature]
            ],
            z=X_test_imputed[
                no_stroke,
                feature_positions[z_feature]
            ],
            mode="markers",
            name="실제 뇌졸중 아님",
            marker=dict(
                color="#4C78A8",
                size=4,
                opacity=0.7
            )
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=X_test_imputed[
                stroke,
                feature_positions[x_feature]
            ],
            y=X_test_imputed[
                stroke,
                feature_positions[y_feature]
            ],
            z=X_test_imputed[
                stroke,
                feature_positions[z_feature]
            ],
            mode="markers",
            name="실제 뇌졸중",
            marker=dict(
                color="#E45756",
                size=5,
                symbol="diamond",
                opacity=0.85
            )
        )
    )

    boundary_configs_3d = [
        (
            "로지스틱 회귀(확률로 답하는 모델)",
            logistic_before,
            "logistic",
            "가중치 조정 전",
            "#F58518"
        ),
        (
            "로지스틱 회귀(확률로 답하는 모델)",
            logistic_after,
            "logistic",
            "가중치 조정 후",
            "#B279A2"
        ),
        (
            "의사결정트리(질문으로 답하는 모델)",
            tree_before,
            "tree",
            "가중치 조정 전",
            "#54A24B"
        ),
        (
            "의사결정트리(질문으로 답하는 모델)",
            tree_after,
            "tree",
            "가중치 조정 후",
            "#72B7B2"
        )
    ]

    for model_name, model, model_kind, state, color in boundary_configs_3d:
        xx, yy, zz, probability_grid = get_3d_boundary(
            model,
            model_kind,
            x_values,
            y_values,
            z_values
        )

        min_probability = float(np.nanmin(probability_grid))
        max_probability = float(np.nanmax(probability_grid))

        if min_probability <= 0.5 <= max_probability:
            fig.add_trace(
                go.Isosurface(
                    x=xx.ravel(),
                    y=yy.ravel(),
                    z=zz.ravel(),
                    value=probability_grid.ravel(),
                    isomin=0.499,
                    isomax=0.501,
                    surface_count=1,
                    opacity=0.22,
                    colorscale=[
                        [0, color],
                        [1, color]
                    ],
                    showscale=False,
                    caps=dict(
                        x_show=False,
                        y_show=False,
                        z_show=False
                    ),
                    name=f"{model_name} {state} 0.5 평면",
                    hoverinfo="skip"
                )
            )
        else:
            st.warning(
                f"{model_name} {state}: "
                "0.5 경계 평면이 현재 그림 범위 밖에 있습니다."
            )

    fig.update_layout(
        scene=dict(
            xaxis_title=FEATURE_NAMES[x_feature],
            yaxis_title=FEATURE_NAMES[y_feature],
            zaxis_title=FEATURE_NAMES[z_feature]
        ),
        height=750,
        title="3차원 테스트 데이터와 0.5 경계 평면",
        legend_title="실제 결과 및 경계 평면"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.caption(
        "3차원 그래프는 마우스로 회전하고 확대할 수 있습니다. "
        "경계 평면은 뇌졸중일 확률이 0.5인 위치를 나타냅니다. "
        "선택한 속성이 4개 이상이면 나머지 속성은 테스트 데이터의 중앙값으로 고정합니다."
    )


# --------------------------------------------------
# 설명
# --------------------------------------------------
st.markdown("### 결과 해석 안내")

st.write(
    "- 가중치 조정 후 재현율이 높아지면 실제 뇌졸중 환자를 더 많이 찾아냈다는 뜻입니다."
)

st.write(
    "- 재현율이 높아져도 헛짚은 사람(FP)이 늘어날 수 있으므로 정밀도도 함께 확인해야 합니다."
)

st.write(
    "- 정확도만으로는 드문 뇌졸중 사례를 잘 찾아내는지 판단하기 어려울 수 있습니다."
)

st.write(
    "- 이 결과는 제공된 데이터와 해당 분할 방식에서의 실습 결과이며, "
    "실제 의료 진단에 사용할 수 있는 모델의 성능을 뜻하지 않습니다."
)
