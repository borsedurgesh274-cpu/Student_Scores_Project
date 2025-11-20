import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import plotly.express as px

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Durgesh Borse - Student Score Predictor",
    layout="wide",
)

# ---------- CUSTOM CSS (GRADIENT + CARDS) ----------
custom_css = """
<style>
/* App background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f172a, #1e293b, #0369a1);
    color: #f9fafb;
}

/* Remove default header background */
[data-testid="stHeader"] {
    background: rgba(0, 0, 0, 0);
}

/* Main title */
.big-title {
    font-size: 3rem;
    font-weight: 800;
    text-align: center;
    color: #e5e7eb;
    margin-bottom: 0.2rem;
}

/* Subtitle */
.sub-title {
    text-align: center;
    color: #d1d5db;
    font-size: 1.1rem;
    margin-bottom: 1.5rem;
}

/* Card style */
.card {
    padding: 1rem 1.5rem;
    border-radius: 1rem;
    background: rgba(15, 23, 42, 0.9);
    border: 1px solid rgba(148, 163, 184, 0.4);
    box-shadow: 0 18px 45px rgba(15, 23, 42, 0.8);
}

/* Metric numbers */
.metric-label {
    font-size: 0.9rem;
    color: #9ca3af;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #e5e7eb;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ---------- TITLE ----------
st.markdown("<div class='big-title'>Durgesh Borse</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-title'>📊 Student Score Prediction Dashboard</div>",
    unsafe_allow_html=True,
)
st.markdown("---")

# ---------- FILE PATHS ----------
DATA_PATH = "Student_Data (1).csv"
MODEL_PATH = "Student_model (3).pkl"

# ---------- LOADERS ----------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_resource
def load_model(path: str):
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model


# ---------- LOAD DATA + MODEL WITH SAFETY CHECKS ----------
data_file = Path(DATA_PATH)
model_file = Path(MODEL_PATH)

if not data_file.exists():
    st.error(f"❌ Could not find data file: {DATA_PATH}")
    st.stop()

if not model_file.exists():
    st.error(f"❌ Could not find model file: {MODEL_PATH}")
    st.stop()

df = load_data(DATA_PATH)
model = load_model(MODEL_PATH)

# ---------- DASHBOARD METRICS ----------
avg_score = df["Score"].mean()
avg_hours = df["Hours_Studied"].mean()
avg_attendance = df["Attendance"].mean()

col_m1, col_m2, col_m3 = st.columns(3)

with col_m1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='metric-label'>Average Score</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='metric-value'>{avg_score:.1f}</div>", unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

with col_m2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='metric-label'>Average Study Hours</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='metric-value'>{avg_hours:.1f} hrs</div>", unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

with col_m3:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='metric-label'>Average Attendance</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='metric-value'>{avg_attendance:.1f} %</div>", unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("")  # small spacing

# ---------- DEFAULT VALUES FOR INPUTS ----------
default_hours = float(df["Hours_Studied"].mean())
default_attendance = float(df["Attendance"].mean())
default_assignments = float(df["Assignments_Submitted"].mean())

# Initialise session_state for reset functionality
if "hours" not in st.session_state:
    st.session_state["hours"] = default_hours
if "attendance" not in st.session_state:
    st.session_state["attendance"] = default_attendance
if "assignments" not in st.session_state:
    st.session_state["assignments"] = default_assignments

# ---------- LAYOUT: INPUTS & CHART ----------
left_col, right_col = st.columns([1, 2])

# ----- LEFT: INPUT FORM -----
with left_col:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🎯 Predict Student Score")

    hours = st.number_input(
        "Hours Studied",
        min_value=0.0,
        max_value=24.0,
        step=0.5,
        key="hours",
    )

    attendance = st.number_input(
        "Attendance (%)",
        min_value=0.0,
        max_value=100.0,
        step=1.0,
        key="attendance",
    )

    assignments = st.number_input(
        "Assignments Submitted",
        min_value=0,
        max_value=20,
        step=1,
        key="assignments",
    )

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        predict_btn = st.button("🚀 Predict Score", use_container_width=True)
    with col_btn2:
        reset_btn = st.button("🔄 Reset Inputs", use_container_width=True)

    # Reset logic
    if reset_btn:
        st.session_state["hours"] = default_hours
        st.session_state["attendance"] = default_attendance
        st.session_state["assignments"] = default_assignments
        st.rerun()

    if predict_btn:
        # Prepare input for model
        X_new = pd.DataFrame(
            {
                "Hours_Studied": [st.session_state["hours"]],
                "Attendance": [st.session_state["attendance"]],
                "Assignments_Submitted": [st.session_state["assignments"]],
            }
        )

        try:
            y_pred = model.predict(X_new)[0]
            st.success(f"✅ Predicted Score: **{y_pred:.2f}**")
        except Exception as e:
            st.error(
                "⚠️ Could not make prediction. "
                "Please check that the model uses columns: 'Hours_Studied', 'Attendance', 'Assignments_Submitted'."
            )
            st.text(f"Error: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

# ----- RIGHT: CHART + DATA TABLE -----
with right_col:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📈 Hours Studied vs Score")

    fig = px.scatter(
        df,
        x="Hours_Studied",
        y="Score",
        size_max=12,
        title="Hours Studied vs Score",
    )
    # Make chart background transparent to match gradient
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.9)",
        font=dict(color="#e5e7eb"),
    )

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("🔍 View Raw Data"):
        st.dataframe(df, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)
