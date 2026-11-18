import os
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓")
FEATURES = ["attendance", "study_hours", "prev_marks", "assignments"]


@st.cache_resource
def train_model():
    # Use your own dataset if data.csv exists (needs the 4 feature columns + "passed" 0/1)
    if os.path.exists("data.csv"):
        df = pd.read_csv("data.csv")
    else:  # otherwise generate a sample dataset
        rng = np.random.default_rng(42)
        n = 1000
        df = pd.DataFrame({
            "attendance": rng.integers(40, 101, n),
            "study_hours": rng.uniform(0, 10, n).round(1),
            "prev_marks": rng.integers(30, 101, n),
            "assignments": rng.integers(0, 11, n),
        })
        score = (0.35 * df.attendance + 4 * df.study_hours
                 + 0.35 * df.prev_marks + 1.5 * df.assignments
                 + rng.normal(0, 5, n))
        df["passed"] = (score >= 70).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df["passed"], test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, acc


model, acc = train_model()

st.title("🎓 Student Performance Predictor")
st.caption(f"Random Forest model | Test accuracy: {acc:.1%}")

attendance = st.slider("Attendance (%)", 0, 100, 75)
study_hours = st.slider("Study hours per day", 0.0, 10.0, 3.0, 0.5)
prev_marks = st.slider("Previous exam marks (%)", 0, 100, 60)
assignments = st.slider("Assignments completed (out of 10)", 0, 10, 7)

if st.button("Predict", type="primary"):
    row = pd.DataFrame([[attendance, study_hours, prev_marks, assignments]], columns=FEATURES)
    prob = model.predict_proba(row)[0][1]
    if prob >= 0.5:
        st.success(f"✅ Likely to PASS ({prob:.0%} confidence)")
    else:
        st.error(f"⚠️ Likely to FAIL (pass chance {prob:.0%})")
    st.progress(float(prob))

with st.expander("Which factors matter most?"):
    imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
    st.bar_chart(imp)
