
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, _tree


# ==================================================
# 페이지 설정
# ==================================================

st.set_page_config(
    page_title="뇌졸중 예측 실습실 | 분류 모델",
    page_icon="🧠",
    layout="wide",
)

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "s200ua1/data/refs/heads/main/stroke.csv"
)

RANDOM_STATE = 42

FEATURE_NAMES = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병",
}


# ==================================================
# 데이터 불러오기
# ==================================================

@st.cache_data
def load_data():
    return pd.read_csv(
        DATA_URL,
        encoding="utf-8",
    )


st.title("🧠 분류 모델")
st.write(
    "로지스틱 회귀와 의사결정트리를 이용해 "
    "뇌졸중 여부를 분류하고 성능을 비교합니다."
)

st.info(
    "이 실습은 공개 데이터를 이용한 학습용 분류 실습입니다. "
    "결과는 의료적 진단이나 개인의 실제 위험 판단에 사용할 수 없습니다."
)

try:
    df = load_data()

except Exception as error:
    st.error("데이터를 불러오지 못했습니다.")
    st.code(str(error))
    st.stop()


required_columns = [
    "id",
    "age",
    "avg_glucose_level",
    "bmi",
    "hypertension",
    "heart_disease",
    "stroke",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    st.error(
        "필수 열이 없습니다: "
        + ", ".join(missing_columns)
    )
    st.stop()


# 숫자형 변환
for column in required_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce",
    )


# 정답이 없는 행 제거
df = df.dropna(
    subset=["stroke"]
).copy()

# 정답은 0 또는 1인 데이터만 사용
df = df[
    df["stroke"].isin([0, 1])
].copy()

df["stroke"] = df["stroke"].astype(int)


# ==================================================
# 속성 선택
# ==================================================

st.subheader("1. 입력 속성 선택")

st.write(
    "분류 모델이 뇌졸중 여부를 판단할 때 사용할 "
    "속성을 선택하세요."
)

feature_options = list(FEATURE_NAMES.keys())

default_features = [
    "age",
    "avg_glucose_level",
    "hypertension",
    "heart_disease",
]

selected_features = st.multiselect(
    "입력 속성",
    options=feature_options,
    default=default_features,
    format_func=lambda x: FEATURE_NAMES[x],
)

if len(selected_features) < 2:
    st.warning(
        "입력 속성을 두 개 이상 선택해야 합니다."
    )
    st.stop()


# ==================================================
# 번호 순 정렬 후 10명씩 묶어 테스트 데이터 구성
# ==================================================

st.subheader("2. 학습 데이터와 테스트 데이터")

# id 순으로 정렬
df = df.sort_values(
    by="id",
    ascending=True,
).reset_index(drop=True)


# 10명씩 묶었을 때 앞의 3명을 테스트 데이터로 지정
row_positions = np.arange(len(df))

test_mask = (
    row_positions % 10 < 3
)

train_df = df.loc[
    ~test_mask
].copy()

test_df = df.loc[
    test_mask
].copy()


# 입력 데이터와 정답
X_train = train_df[
    selected_features
].copy()

X_test = test_df[
    selected_features
].copy()

y_train = train_df[
    "stroke"
].copy()

y_test = test_df[
    "stroke"
].copy()


# 결측치 대체는 학습 데이터에만 fit
# BMI를 선택한 경우에만 BMI의 결측치가 처리됨
imputer = SimpleImputer(
    strategy="median"
)

X_train_imputed = imputer.fit_transform(
    X_train
)

X_test_imputed = imputer.transform(
    X_test
)


# 학습 데이터로만 표준화 기준을 계산
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train_imputed
)

X_test_scaled = scaler.transform(
    X_test_imputed
)


# 테스트 데이터 개수 확인
col1, col2, col3 = st.columns(3)

col1.metric(
    "전체 데이터",
    f"{len(df):,}명",
)

col2.metric(
    "훈련 데이터",
    f"{len(train_df):,}명",
)

col3.metric(
    "테스트 데이터",
    f"{len(test_df):,}명",
)

if len(test_df) != 1533:
    st.warning(
        f"현재 데이터의 테스트 인원은 {len(test_df):,}명입니다. "
        "데이터 행 수가 달라지면 테스트 인원도 달라질 수 있습니다."
    )


# ==================================================
# 모델 학습
# ==================================================

st.subheader("3. 분류 모델 학습")


# 로지스틱 회귀
logistic_model = LogisticRegression(
    random_state=RANDOM_STATE,
    max_iter=1000,
)

logistic_model.fit(
    X_train_scaled,
    y_train,
)


# 의사결정트리
tree_model = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    random_state=RANDOM_STATE,
)

tree_model.fit(
    X_train_imputed,
    y_train,
)


# ==================================================
# 정확도 계산
# ==================================================

# 로지스틱 회귀
logistic_train_pred = logistic_model.predict(
    X_train_scaled
)

logistic_test_pred = logistic_model.predict(
    X_test_scaled
)

logistic_train_accuracy = accuracy_score(
    y_train,
    logistic_train_pred,
)

logistic_test_accuracy = accuracy_score(
    y_test,
    logistic_test_pred,
)


# 의사결정트리
tree_train_pred = tree_model.predict(
    X_train_imputed
)

tree_test_pred = tree_model.predict(
    X_test_imputed
)

tree_train_accuracy = accuracy_score(
    y_train,
    tree_train_pred,
)

tree_test_accuracy = accuracy_score(
    y_test,
    tree_test_pred,
)


# ==================================================
# 다수 클래스 기준 모델
# ==================================================

# 훈련 데이터에서 더 많은 정답을 항상 선택
majority_class = int(
    y_train.value_counts().idxmax()
)

majority_train_pred = np.full(
    len(y_train),
    majority_class,
)

majority_test_pred = np.full(
    len(y_test),
    majority_class,
)

majority_train_accuracy = accuracy_score(
    y_train,
    majority_train_pred,
)

majority_test_accuracy = accuracy_score(
    y_test,
    majority_test_pred,
)


# ==================================================
# 정확도 카드
# ==================================================

st.subheader("4. 모델 정확도 비교")

st.caption(
    "큰 숫자는 테스트 데이터 정확도이며, "
    "아래에는 훈련 데이터와 테스트 데이터 정확도를 함께 표시합니다."
)


def show_accuracy_card(
    title,
    train_accuracy,
    test_accuracy,
):

    with st.container(border=True):

        st.markdown(f"**{title}**")

        st.metric(
            label="테스트 데이터 정확도",
            value=f"{test_accuracy:.2%}",
        )

        small_col1, small_col2 = st.columns(2)

        small_col1.caption(
            f"훈련 데이터: {train_accuracy:.2%}"
        )

        small_col2.caption(
            f"테스트 데이터: {test_accuracy:.2%}"
        )


card1, card2, card3 = st.columns(3)

with card1:

    show_accuracy_card(
        "로지스틱 회귀(확률로 답하는 모델)",
        logistic_train_accuracy,
        logistic_test_accuracy,
    )

with card2:

    show_accuracy_card(
        "의사결정트리(질문으로 답하는 모델)",
        tree_train_accuracy,
        tree_test_accuracy,
    )

with card3:

    show_accuracy_card(
        "입력을 보지 않고 많은 쪽으로 답하는 모델",
        majority_train_accuracy,
        majority_test_accuracy,
    )


st.divider()


# ==================================================
# 시각화에 사용할 두 속성 선택
# ==================================================

st.subheader("5. 두 속성으로 분류 결과 시각화")

st.write(
    "가로축과 세로축에 사용할 속성을 선택하세요. "
    "두 축에 사용하지 않는 속성은 테스트 데이터의 중앙값에 고정합니다."
)

axis_col1, axis_col2 = st.columns(2)

with axis_col1:

    x_feature = st.selectbox(
        "가로축",
        options=selected_features,
        index=0,
        format_func=lambda x: FEATURE_NAMES[x],
    )

with axis_col2:

    y_feature_options = [
        feature
        for feature in selected_features
        if feature != x_feature
    ]

    y_feature = st.selectbox(
        "세로축",
        options=y_feature_options,
        index=0,
        format_func=lambda x: FEATURE_NAMES[x],
    )


# 테스트 데이터의 중앙값으로 고정할 속성
fixed_features = [
    feature
    for feature in selected_features
    if feature not in [x_feature, y_feature]
]


# 중앙값 계산은 테스트 데이터 기준
fixed_values = {}

for feature in fixed_features:

    fixed_values[feature] = float(
        X_test[feature].median()
    )


if fixed_features:

    fixed_text = ", ".join(
        [
            f"{FEATURE_NAMES[feature]} = "
            f"{fixed_values[feature]:.3f}"
            for feature in fixed_features
        ]
    )

    st.write(
        "그림에서 고정한 속성의 값: "
        + fixed_text
    )

else:

    st.write(
        "선택한 모든 속성이 가로축과 세로축에 사용됩니다."
    )


# ==================================================
# 분류 영역과 0.5 경계선
# ==================================================

# 시각화 범위
x_min = float(
    X_test[x_feature].min()
)

x_max = float(
    X_test[x_feature].max()
)

y_min = float(
    X_test[y_feature].min()
)

y_max = float(
    X_test[y_feature].max()
)


# 범위가 동일한 경우 작은 여백 추가
if x_min == x_max:

    x_min -= 1
    x_max += 1

if y_min == y_max:

    y_min -= 1
    y_max += 1


x_margin = (x_max - x_min) * 0.05
y_margin = (y_max - y_min) * 0.05

x_min -= x_margin
x_max += x_margin

y_min -= y_margin
y_max += y_margin


xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 180),
    np.linspace(y_min, y_max, 180),
)


# 전체 선택 속성의 입력 행렬 생성
grid_data = pd.DataFrame(
    np.zeros(
        (xx.size, len(selected_features))
    ),
    columns=selected_features,
)


# 축 속성 입력
grid_data[x_feature] = xx.ravel()
grid_data[y_feature] = yy.ravel()


# 나머지 속성은 테스트 중앙값으로 고정
for feature in fixed_features:

    grid_data[feature] = fixed_values[
        feature
    ]


# 로지스틱 회귀에 사용할 표준화 입력
grid_imputed = imputer.transform(
    grid_data
)

grid_scaled = scaler.transform(
    grid_imputed
)


# 확률 계산
logistic_grid_probability = (
    logistic_model.predict_proba(
        grid_scaled
    )[:, 1]
)


logistic_grid_probability = (
    logistic_grid_probability.reshape(
        xx.shape
    )
)


# 의사결정트리 영역
tree_grid_prediction = (
    tree_model.predict(
        grid_imputed
    )
)

tree_grid_prediction = (
    tree_grid_prediction.reshape(
        xx.shape
    )
)


# ==================================================
# 산점도
# ==================================================

st.subheader(
    "테스트 데이터와 로지스틱 회귀의 0.5 경계선"
)

plot_df = X_test.copy()

plot_df["실제 뇌졸중 여부"] = (
    y_test.map(
        {
            0: "뇌졸중 아님",
            1: "뇌졸중",
        }
    )
)

plot_df["예측 확률"] = (
    logistic_model.predict_proba(
        X_test_scaled
    )[:, 1]
)


fig_scatter = go.Figure()


# 의사결정트리 분류 영역
fig_scatter.add_trace(
    go.Contour(
        x=np.linspace(
            x_min,
            x_max,
            180,
        ),
        y=np.linspace(
            y_min,
            y_max,
            180,
        ),
        z=tree_grid_prediction,
        showscale=False,
        opacity=0.18,
        colorscale=[
            [0, "rgb(80, 140, 220)"],
            [1, "rgb(240, 100, 100)"],
        ],
        contours=dict(
            coloring="fill",
            showlines=False,
        ),
        hoverinfo="skip",
        name="의사결정트리 영역",
    )
)


# 로지스틱 회귀의 0.5 경계선
fig_scatter.add_trace(
    go.Contour(
        x=np.linspace(
            x_min,
            x_max,
            180,
        ),
        y=np.linspace(
            y_min,
            y_max,
            180,
        ),
        z=logistic_grid_probability,
        showscale=False,
        contours=dict(
            start=0.5,
            end=0.5,
            size=1,
            coloring="lines",
            showlabels=False,
        ),
        line=dict(
            color="black",
            width=3,
        ),
        hoverinfo="skip",
        name="로지스틱 회귀 0.5 경계선",
    )
)


# 실제 뇌졸중 여부별 테스트 점
for label, color in [
    ("뇌졸중 아님", "blue"),
    ("뇌졸중", "red"),
]:

    subset = plot_df[
        plot_df["실제 뇌졸중 여부"] == label
    ]

    fig_scatter.add_trace(
        go.Scatter(
            x=subset[x_feature],
            y=subset[y_feature],
            mode="markers",
            name=label,
            marker=dict(
                color=color,
                size=7,
                opacity=0.75,
            ),
            customdata=subset[
                ["예측 확률"]
            ],
            hovertemplate=(
                f"{FEATURE_NAMES[x_feature]}: %{{x}}<br>"
                f"{FEATURE_NAMES[y_feature]}: %{{y}}<br>"
                "실제 결과: "
                + label
                + "<br>뇌졸중 예측 확률: %{customdata[0]:.3f}"
                + "<extra></extra>"
            ),
        )
    )


fig_scatter.update_layout(
    title="테스트 데이터의 실제 결과와 모델 경계",
    xaxis_title=FEATURE_NAMES[x_feature],
    yaxis_title=FEATURE_NAMES[y_feature],
    legend_title="실제 결과",
    height=650,
)

st.plotly_chart(
    fig_scatter,
    use_container_width=True,
)


# 경계선이 그림 범위 안에 있는지 확인
boundary_values = (
    logistic_grid_probability - 0.5
)

boundary_crosses = (
    np.nanmin(boundary_values) <= 0
    and np.nanmax(boundary_values) >= 0
)

if boundary_crosses:

    st.write(
        "로지스틱 회귀의 0.5 경계선은 "
        "현재 그림 범위 안에 있습니다."
    )

else:

    st.write(
        "로지스틱 회귀의 0.5 경계선은 "
        "현재 그림 범위 밖에 있습니다."
    )


st.divider()


# ==================================================
# 의사결정트리 가지 그림
# ==================================================

st.subheader(
    "6. 의사결정트리가 던지는 질문"
)

st.write(
    "각 마디에는 그 마디에 도달한 훈련용 사람 수, "
    "실제 뇌졸중인 사람 수, 뇌졸중 비율이 표시됩니다."
)


tree = tree_model.tree_

dot_lines = [
    "digraph DecisionTree {",
    'graph [rankdir=TB, bgcolor="white"];',
    'node [shape=box, style="filled,rounded", fontname="Arial"];',
    'edge [fontname="Arial"];',
]


# 노드별 문장 생성
node_labels = {}

leaf_nodes = []

for node_id in range(tree.node_count):

    samples = int(
        tree.n_node_samples[node_id]
    )

    values = tree.value[
        node_id
    ][0]

    negative_count = int(
        values[0]
    )

    positive_count = int(
        values[1]
    )

    total_count = (
        negative_count
        + positive_count
    )

    positive_rate = (
        positive_count / total_count * 100
        if total_count > 0
        else 0
    )


    is_leaf = (
        tree.children_left[node_id]
        == _tree.TREE_LEAF
    )


    if is_leaf:

        predicted_class = int(
            tree_model.classes_[
                np.argmax(values)
            ]
        )

        if predicted_class == 1:

            answer_text = "답: 뇌졸중"
            node_color = "#F5A3A3"

        else:

            answer_text = "답: 뇌졸중 아님"
            node_color = "#A8C8F0"

        leaf_nodes.append(
            predicted_class
        )

        label = (
            f"마디 {node_id}\\n"
            f"훈련용 사람: {samples}명\\n"
            f"실제 뇌졸중: {positive_count}명\\n"
            f"뇌졸중 비율: {positive_rate:.1f}%\\n"
            f"{answer_text}"
        )

    else:

        feature_index = int(
            tree.feature[node_id]
        )

        feature_name = selected_features[
            feature_index
        ]

        threshold = float(
            tree.threshold[node_id]
        )

        question = (
            f"{FEATURE_NAMES[feature_name]} "
            f"≤ {threshold:.3f}?"
        )

        label = (
            f"마디 {node_id}\\n"
            f"{question}\\n"
            f"훈련용 사람: {samples}명\\n"
            f"실제 뇌졸중: {positive_count}명\\n"
            f"뇌졸중 비율: {positive_rate:.1f}%"
        )

        node_color = "#FFF2CC"


    node_labels[node_id] = label

    dot_lines.append(
        f'{node_id} [label="{label}", '
        f'fillcolor="{node_color}"];'
    )


# 가지 연결
for node_id in range(tree.node_count):

    left_child = int(
        tree.children_left[node_id]
    )

    right_child = int(
        tree.children_right[node_id]
    )


    if left_child == _tree.TREE_LEAF:

        continue


    # 왼쪽 가지: 조건을 만족하는 예
    dot_lines.append(
        f'{node_id} -> {left_child} '
        '[label="예"];'
    )


    # 오른쪽 가지: 조건을 만족하지 않는 아니요
    dot_lines.append(
        f'{node_id} -> {right_child} '
        '[label="아니요"];'
    )


dot_lines.append("}")

dot_string = "\n".join(
    dot_lines
)


st.graphviz_chart(
    dot_string,
    use_container_width=True,
)


# ==================================================
# 나무 요약
# ==================================================

st.subheader(
    "의사결정트리 요약"
)


# 답을 내는 마디 개수
leaf_count = int(
    np.sum(
        tree.children_left
        == _tree.TREE_LEAF
    )
)


# 뇌졸중 아님으로 답하는 마디 개수
negative_leaf_count = int(
    sum(
        1
        for node_id in range(
            tree.node_count
        )
        if (
            tree.children_left[node_id]
            == _tree.TREE_LEAF
            and int(
                tree_model.classes_[
                    np.argmax(
                        tree.value[node_id][0]
                    )
                ]
            ) == 0
        )
    )
)


# 실제로 질문한 속성
asked_feature_indices = (
    tree.feature[
        tree.feature >= 0
    ]
)

asked_features = sorted(
    set(
        selected_features[index]
        for index in asked_feature_indices
    )
)

asked_feature_names = [
    FEATURE_NAMES[feature]
    for feature in asked_features
]


st.write(
    f"답을 내는 마디는 모두 **{leaf_count}칸**이며, "
    f"그중 **{negative_leaf_count}칸**이 "
    "‘뇌졸중 아님’이라고 답합니다."
)

st.write(
    "이 나무가 실제로 질문한 속성: "
    + (
        ", ".join(asked_feature_names)
        if asked_feature_names
        else "질문 없음"
    )
)


st.divider()

st.caption(
    "뇌졸중 예측 실습실 | 분류 모델 페이지"
)
