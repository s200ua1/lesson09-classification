import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

# 페이지 설정: 브라우저 탭 제목과 아이콘
st.set_page_config(
    page_title="뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide",
)

DATA_URL = "https://raw.githubusercontent.com/s200ua1/data/refs/heads/main/stroke.csv"
RECORD_DIR = Path("내_기록")
RECORD_FILE = RECORD_DIR / "기록.json"

COLUMNS = [
    "id", "gender", "age", "hypertension", "heart_disease", "ever_married",
    "work_type", "Residence_type", "avg_glucose_level", "bmi",
    "smoking_status", "stroke"
]

# 교재를 보고 직접 채울 수 있도록 우리말 뜻은 처음에 비워 둠
DEFAULT_RECORD = {
    "meanings": {column: "" for column in COLUMNS},
    "source": "",
}


def load_record():
    """저장된 우리말 뜻과 출처를 읽는다."""
    try:
        if RECORD_FILE.exists():
            with RECORD_FILE.open("r", encoding="utf-8") as file:
                saved = json.load(file)
            meanings = DEFAULT_RECORD["meanings"].copy()
            meanings.update(saved.get("meanings", {}))
            return {"meanings": meanings, "source": saved.get("source", "")}
    except (OSError, json.JSONDecodeError, TypeError):
        st.warning("저장된 기록을 읽지 못해 빈 기록으로 시작합니다.")
    return {
        "meanings": DEFAULT_RECORD["meanings"].copy(),
        "source": "",
    }


def save_record(meanings, source):
    """입력 내용을 JSON 파일로 저장한다."""
    try:
        RECORD_DIR.mkdir(parents=True, exist_ok=True)
        payload = {"meanings": meanings, "source": source}
        with RECORD_FILE.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
        return True, ""
    except OSError as error:
        return False, str(error)


@st.cache_data
def load_data(url):
    """UTF-8 CSV를 읽고 컬럼 이름을 확인한다."""
    data = pd.read_csv(url, encoding="utf-8")
    missing_columns = [column for column in COLUMNS if column not in data.columns]
    if missing_columns:
        raise ValueError("필수 열이 없습니다: " + ", ".join(missing_columns))
    return data


# 데이터 불러오기
try:
    df = load_data(DATA_URL)
except Exception as error:
    st.set_page_config(page_title="뇌졸중 예측 실습실", page_icon="🧠", layout="wide")
    st.title("🧠 뇌졸중 예측 실습실")
    st.error("데이터를 불러오지 못했습니다. 인터넷 연결과 CSV 주소를 확인해 주세요.")
    st.code(str(error))
    st.stop()

record = load_record()

# 제목
st.title("🧠 뇌졸중 예측 실습실")
st.write(
    "이 실습실은 뇌졸중 관련 공개 데이터를 살펴보고, 각 열이 뜻하는 바와 "
    "데이터의 구조를 직접 정리하는 공간입니다."
)
st.info(
    "주의: 이 데이터와 앱은 학습·실습용입니다. 아래의 통계나 예측 결과는 "
    "의학적 진단이나 개인의 실제 뇌졸중 위험 판단에 사용할 수 없습니다."
)

# 숫자 카드 4개
st.subheader("데이터 한눈에 보기")
total_people = len(df)
column_count = len(df.columns)
stroke_count = int((pd.to_numeric(df["stroke"], errors="coerce") == 1).sum())
stroke_rate = (stroke_count / total_people * 100) if total_people else 0.0

card1, card2, card3, card4 = st.columns(4)
card1.metric("전체 사람 수", f"{total_people:,}명")
card2.metric("열 개수", f"{column_count:,}개")
card3.metric("stroke가 1인 사람 수", f"{stroke_count:,}명")
card4.metric("stroke가 1인 비율", f"{stroke_rate:.2f}%")

st.divider()

# 열 설명 표: 우리말 뜻 칸만 직접 편집할 수 있게 구성
st.subheader("열 이름과 데이터 구조")
st.caption("표의 ‘우리말 뜻’ 칸을 교재를 참고해 채운 다음, 아래 저장 버튼을 눌러 주세요.")

kind_rows = []
for column in df.columns:
    series = df[column]
    non_null_values = series.dropna()
    if pd.api.types.is_numeric_dtype(series):
        value_kind = f"숫자형 · 고유값 {series.nunique(dropna=True):,}개"
    else:
        unique_values = non_null_values.astype(str).unique().tolist()
        if len(unique_values) <= 8:
            value_kind = "문자형 · " + (", ".join(unique_values) if unique_values else "값 없음")
        else:
            value_kind = f"문자형 · 고유값 {len(unique_values):,}개"
    kind_rows.append({
        "열 이름": column,
        "우리말 뜻": record["meanings"].get(column, ""),
        "값의 종류": value_kind,
        "빈 값 개수": int(series.isna().sum()),
    })

info_df = pd.DataFrame(kind_rows)
edited_info_df = st.data_editor(
    info_df,
    use_container_width=True,
    hide_index=True,
    num_rows="fixed",
    disabled=["열 이름", "값의 종류", "빈 값 개수"],
    column_config={
        "열 이름": st.column_config.TextColumn("열 이름"),
        "우리말 뜻": st.column_config.TextColumn(
            "우리말 뜻",
            help="교재를 참고해 각 열의 뜻을 입력하세요.",
            width="large",
        ),
        "값의 종류": st.column_config.TextColumn("값의 종류"),
        "빈 값 개수": st.column_config.NumberColumn("빈 값 개수", format="%d"),
    },
    key="column_meanings_editor",
)
edited_meanings = {
    str(row["열 이름"]): str(row["우리말 뜻"]) if pd.notna(row["우리말 뜻"]) else ""
    for _, row in edited_info_df.iterrows()
}

source_text = st.text_area(
    "데이터 출처",
    value=record.get("source", ""),
    placeholder="교재에 적힌 데이터 출처를 입력하세요.",
    height=90,
    key="source_text",
)

if st.button("💾 우리말 뜻과 출처 저장", type="primary"):
    success, message = save_record(edited_meanings, source_text)
    if success:
        st.success(f"기록을 저장했습니다: {RECORD_FILE.as_posix()}")
    else:
        st.error(
            "파일로 저장하지 못했습니다. 실행 환경의 파일 쓰기 권한을 확인해 주세요. "
            f"오류: {message}"
        )

st.divider()

# 처음 다섯 줄 원본
st.subheader("원본 데이터 처음 5행")
st.caption("아래 표는 데이터의 첫 다섯 행을 원래 열 이름과 값으로 표시합니다.")
st.dataframe(df.head(5), use_container_width=True, hide_index=True)

st.divider()
st.caption("참고: 이 앱은 데이터 탐색과 교육용 실습을 위한 것이며 의료적 판단을 대신하지 않습니다.")
