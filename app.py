from io import BytesIO
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
COLUMNS = ["feedback_id", "department", "text"]
ALL_DEPARTMENTS = "Все подразделения"
ALL_TOPICS = "Все темы"
REVIEW_TOPIC = "Требуется ручной разбор"


@st.cache_resource
def load_model():
    return joblib.load(ROOT / "model.joblib")


def classify_feedback(content, model):
    try:
        feedback = pd.read_csv(
            BytesIO(content),
            encoding="utf-8-sig",
            dtype=str,
            keep_default_na=False,
        )
    except UnicodeDecodeError as error:
        raise ValueError("Сохраните CSV в кодировке UTF-8.") from error
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as error:
        raise ValueError(
            "Проверьте формат CSV и используйте запятую как разделитель."
        ) from error

    missing = [column for column in COLUMNS if column not in feedback.columns]
    if missing:
        raise ValueError(f"В CSV не хватает столбцов: {', '.join(missing)}.")
    if feedback.empty:
        raise ValueError("Добавьте в CSV хотя бы один отзыв.")

    feedback = feedback.loc[:, COLUMNS].copy()
    for column in COLUMNS:
        feedback[column] = feedback[column].str.strip()

    if feedback.eq("").any().any():
        raise ValueError("Заполните feedback_id, department и text у каждого отзыва.")
    if feedback["feedback_id"].duplicated().any():
        raise ValueError("У каждого отзыва должен быть уникальный feedback_id.")

    review_threshold = 0.21
    probabilities = model.predict_proba(feedback["text"])
    best_scores = probabilities.max(axis=1)
    best_classes = probabilities.argmax(axis=1)

    feedback["prediction"] = model.classes_[best_classes]
    feedback["topic"] = feedback["prediction"].mask(
        best_scores < review_threshold,
        REVIEW_TOPIC,
    )
    return feedback

st.set_page_config(page_title="Новичок", layout="wide")
st.title("Новичок")
st.write("Отзывы о первой неделе работы")

try:
    model = load_model()
except (OSError, ValueError):
    st.error("Не удалось открыть модель. Проверьте, что model.joblib лежит рядом с app.py.")
    st.stop()

topics = [*model.classes_, REVIEW_TOPIC]
upload_column, demo_column = st.columns([2, 1], vertical_alignment="center")
with upload_column:
    uploaded = st.file_uploader("CSV с отзывами", type=["csv"], key="upload")
    analyze = st.button("Разобрать CSV", key="analyze_upload", disabled=uploaded is None)
with demo_column:
    load_demo = st.button("Загрузить пример", key="load_demo", type="primary")
    st.caption("32 вымышленных отзыва из четырёх подразделений.")
st.caption("Столбцы CSV: feedback_id, department, text. Кодировка UTF-8, разделитель — запятая.")

if load_demo or analyze:
    try:
        if load_demo:
            content = (ROOT / "data" / "demo.csv").read_bytes()
            source = "Демонстрационные данные"
        else:
            content = uploaded.getvalue()
            source = uploaded.name
        feedback = classify_feedback(content, model)
    except OSError:
        st.error("Файл примера data/demo.csv недоступен.")
    except ValueError as error:
        st.error(str(error))
    else:
        st.session_state.analysis = feedback
        st.session_state.source = source
        st.session_state.department = ALL_DEPARTMENTS
        st.session_state.topic_filter = ALL_TOPICS
        st.session_state.pop("feedback", None)
        st.session_state.pop("saved_feedback", None)

if "analysis" not in st.session_state:
    st.stop()

feedback = st.session_state.analysis
st.caption(f"Источник: {st.session_state.source}")
departments = [ALL_DEPARTMENTS, *sorted(feedback["department"].unique())]
department = st.selectbox("Подразделение", departments, key="department")
department_feedback = feedback
if department != ALL_DEPARTMENTS:
    department_feedback = feedback.loc[feedback["department"].eq(department)]

st.subheader("Сводка по темам")
st.metric("Отзывов в выбранном подразделении", len(department_feedback))
counts = department_feedback["topic"].value_counts().reindex(topics, fill_value=0)
summary = counts.rename("Отзывов").rename_axis("Тема").reset_index()
st.dataframe(summary, hide_index=True, width="stretch")

st.subheader("Исходные отзывы")
if "saved_feedback" in st.session_state:
    st.success(f"Тема отзыва {st.session_state.pop('saved_feedback')} обновлена.")
topic_filter = st.selectbox("Тема", [ALL_TOPICS, *topics], key="topic_filter")
visible = department_feedback
if topic_filter != ALL_TOPICS:
    visible = visible.loc[visible["topic"].eq(topic_filter)]

if visible.empty:
    st.info("По выбранной теме пока нет отзывов.")
    st.stop()

st.dataframe(
    visible.rename(columns={
        "feedback_id": "ID",
        "department": "Подразделение",
        "text": "Отзыв",
        "prediction": "Тема модели",
        "topic": "Тема в сводке",
    }),
    hide_index=True,
    width="stretch",
)

feedback_id = st.selectbox("Отзыв для проверки", visible["feedback_id"], key="feedback")
row = visible.loc[visible["feedback_id"].eq(feedback_id)].iloc[0]
st.text(row["text"])
with st.form(f"edit_{feedback_id}"):
    category = st.selectbox(
        "Тема отзыва",
        topics,
        index=topics.index(row["topic"]),
        key=f"category_{feedback_id}",
    )
    saved = st.form_submit_button("Сохранить тему")

if saved and category != row["topic"]:
    feedback.loc[feedback["feedback_id"].eq(feedback_id), "topic"] = category
    st.session_state.saved_feedback = feedback_id
    st.rerun()
