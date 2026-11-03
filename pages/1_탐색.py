
import pandas as pd
import plotly.express as px
import streamlit as st

# 페이지 설정
st.set_page_config(
    page_title="뇌졸중 예측 실습실 | 탐색",
    page_icon="🧠",
    layout="wide",
)

# 데이터 주소
DATA_URL = (
    "https://raw.githubusercontent.com/"
    "s200ua1/data/refs/heads/main/stroke.csv"
)


# UTF-8 데이터 불러오기
@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


# 데이터 로딩
st.title("🔎 데이터 탐색")
st.write(
    "뇌졸중 데이터의 분포와 그룹별 차이를 "
    "그래프로 살펴보는 페이지입니다."
)

st.caption(
    "학습용 데이터 탐색 결과이며, "
    "의료적 진단이나 개인의 위험 판단에 사용할 수 없습니다."
)

try:
    df = load_data()

except Exception as error:
    st.error("데이터를 불러오지 못했습니다.")
    st.code(str(error))
    st.stop()


# 필요한 열 확인
required_columns = [
    "age",
    "avg_glucose_level",
    "stroke",
    "hypertension",
    "heart_disease",
    "bmi",
    "smoking_status",
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


# 숫자형 열 변환
numeric_columns = [
    "age",
    "avg_glucose_level",
    "stroke",
    "hypertension",
    "heart_disease",
    "bmi",
]

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce",
    )


# 뇌졸중 그룹 이름
stroke_labels = {
    0: "뇌졸중 없음",
    1: "뇌졸중 있음",
}

df["stroke_group"] = df["stroke"].map(
    stroke_labels
)


# ==================================================
# 1. 나이와 평균 혈당의 분포
# ==================================================

st.subheader("1. 나이와 평균 혈당의 분포")

col1, col2 = st.columns(2)

with col1:

    age_data = df.dropna(
        subset=["age"]
    )

    fig_age = px.histogram(
        age_data,
        x="age",
        nbins=30,
        title="나이 분포",
        labels={
            "age": "나이",
            "count": "사람 수",
        },
    )

    fig_age.update_layout(
        yaxis_title="사람 수",
        xaxis_title="나이",
    )

    st.plotly_chart(
        fig_age,
        use_container_width=True,
    )


with col2:

    glucose_data = df.dropna(
        subset=["avg_glucose_level"]
    )

    fig_glucose = px.histogram(
        glucose_data,
        x="avg_glucose_level",
        nbins=30,
        title="평균 혈당 분포",
        labels={
            "avg_glucose_level": "평균 혈당",
            "count": "사람 수",
        },
    )

    fig_glucose.update_layout(
        yaxis_title="사람 수",
        xaxis_title="평균 혈당",
    )

    st.plotly_chart(
        fig_glucose,
        use_container_width=True,
    )


st.divider()


# ==================================================
# 2. 뇌졸중 유무에 따른 나이와 평균 혈당 비교
# ==================================================

st.subheader(
    "2. 뇌졸중 유무에 따른 나이와 평균 혈당 비교"
)

group_df = df[
    df["stroke"].isin([0, 1])
].copy()


col1, col2 = st.columns(2)


# 나이 상자그림
with col1:

    age_box_data = group_df.dropna(
        subset=["age", "stroke_group"]
    )

    fig_age_box = px.box(
        age_box_data,
        x="stroke_group",
        y="age",
        color="stroke_group",
        title="그룹별 나이 상자그림",
        labels={
            "stroke_group": "뇌졸중 여부",
            "age": "나이",
        },
        category_orders={
            "stroke_group": [
                "뇌졸중 없음",
                "뇌졸중 있음",
            ]
        },
    )

    fig_age_box.update_layout(
        showlegend=False,
        xaxis_title="뇌졸중 여부",
        yaxis_title="나이",
    )

    st.plotly_chart(
        fig_age_box,
        use_container_width=True,
    )


# 평균 혈당 상자그림
with col2:

    glucose_box_data = group_df.dropna(
        subset=[
            "avg_glucose_level",
            "stroke_group",
        ]
    )

    fig_glucose_box = px.box(
        glucose_box_data,
        x="stroke_group",
        y="avg_glucose_level",
        color="stroke_group",
        title="그룹별 평균 혈당 상자그림",
        labels={
            "stroke_group": "뇌졸중 여부",
            "avg_glucose_level": "평균 혈당",
        },
        category_orders={
            "stroke_group": [
                "뇌졸중 없음",
                "뇌졸중 있음",
            ]
        },
    )

    fig_glucose_box.update_layout(
        showlegend=False,
        xaxis_title="뇌졸중 여부",
        yaxis_title="평균 혈당",
    )

    st.plotly_chart(
        fig_glucose_box,
        use_container_width=True,
    )


# 두 그룹의 평균값 표
st.markdown("**두 그룹의 평균값 비교**")

mean_table = (
    group_df
    .groupby("stroke_group")[
        ["age", "avg_glucose_level"]
    ]
    .mean()
    .reindex([
        "뇌졸중 없음",
        "뇌졸중 있음",
    ])
)

mean_table = mean_table.rename(
    columns={
        "age": "평균 나이",
        "avg_glucose_level": "평균 혈당",
    }
)

mean_table = mean_table.round(2)

st.dataframe(
    mean_table,
    use_container_width=True,
)


st.divider()


# ==================================================
# 3. 고혈압과 심장병 여부에 따른 뇌졸중 비율
# ==================================================

st.subheader(
    "3. 고혈압과 심장병 여부에 따른 뇌졸중 비율"
)


def make_rate_chart(
    data,
    feature,
    title,
    labels,
):

    valid_data = data[
        data[feature].isin([0, 1])
        & data["stroke"].isin([0, 1])
    ].copy()

    valid_data["상태"] = (
        valid_data[feature].map(labels)
    )

    rates = (
        valid_data
        .groupby("상태")["stroke"]
        .mean()
        .mul(100)
        .reset_index(name="뇌졸중 비율(%)")
    )

    fig = px.bar(
        rates,
        x="상태",
        y="뇌졸중 비율(%)",
        title=title,
        text=rates[
            "뇌졸중 비율(%)"
        ].map(
            lambda value: f"{value:.2f}%"
        ),
        labels={
            "상태": "상태",
        },
    )

    fig.update_traces(
        textposition="outside"
    )

    fig.update_layout(
        yaxis_title="뇌졸중 비율",
        xaxis_title="",
        yaxis_ticksuffix="%",
    )

    return fig


# 고혈압
hypertension_labels = {
    0: "고혈압 없음",
    1: "고혈압 있음",
}

# 심장병
heart_labels = {
    0: "심장병 없음",
    1: "심장병 있음",
}


col1, col2 = st.columns(2)

with col1:

    fig_hypertension = make_rate_chart(
        df,
        "hypertension",
        "고혈압 여부별 뇌졸중 비율",
        hypertension_labels,
    )

    st.plotly_chart(
        fig_hypertension,
        use_container_width=True,
    )


with col2:

    fig_heart = make_rate_chart(
        df,
        "heart_disease",
        "심장병 여부별 뇌졸중 비율",
        heart_labels,
    )

    st.plotly_chart(
        fig_heart,
        use_container_width=True,
    )


st.divider()


# ==================================================
# 4. BMI 빈 값이 있는 사람들의 뇌졸중 비율
# ==================================================

st.subheader(
    "4. BMI 빈 값이 있는 사람들의 뇌졸중 비율"
)


# BMI가 비어 있는 사람
missing_bmi = df[
    df["bmi"].isna()
]


# 뇌졸중 값이 유효한 전체 사람
valid_stroke = df[
    df["stroke"].isin([0, 1])
]


# BMI 빈 값이면서 뇌졸중 값도 유효한 사람
missing_bmi_stroke = missing_bmi[
    missing_bmi["stroke"].isin([0, 1])
]


# BMI 빈 값 그룹의 뇌졸중 비율
if len(missing_bmi_stroke) > 0:

    missing_bmi_rate = (
        missing_bmi_stroke["stroke"].mean()
        * 100
    )

else:

    missing_bmi_rate = float("nan")


# 전체 뇌졸중 비율
if len(valid_stroke) > 0:

    overall_rate = (
        valid_stroke["stroke"].mean()
        * 100
    )

else:

    overall_rate = float("nan")


# 비교 표
bmi_comparison = pd.DataFrame(
    [
        {
            "대상": "BMI가 비어 있는 사람",
            "사람 수": len(missing_bmi),
            "뇌졸중 비율(%)": missing_bmi_rate,
        },
        {
            "대상": "전체 사람",
            "사람 수": len(df),
            "뇌졸중 비율(%)": overall_rate,
        },
    ]
)


bmi_comparison[
    "뇌졸중 비율(%)"
] = bmi_comparison[
    "뇌졸중 비율(%)"
].round(2)


st.dataframe(
    bmi_comparison,
    use_container_width=True,
    hide_index=True,
)


st.divider()


# ==================================================
# 5. 흡연 상태별 사람 수
# ==================================================

st.subheader(
    "5. 흡연 상태별 사람 수"
)


smoking_counts = (
    df["smoking_status"]
    .fillna("빈 값")
    .astype(str)
    .value_counts()
    .rename_axis("흡연 상태")
    .reset_index(name="사람 수")
)


st.dataframe(
    smoking_counts,
    use_container_width=True,
    hide_index=True,
)


# 페이지 하단
st.caption(
    "뇌졸중 예측 실습실 | 데이터 탐색 페이지"
)
