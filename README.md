# ML-Based Student Performance Predictor

A Streamlit web app that predicts whether a student will pass, based on attendance, study hours, previous marks and assignments completed.

## Tech Stack
Python, pandas, scikit-learn (Random Forest), Streamlit

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Dataset
Uses a generated sample dataset by default. To use real data, add a `data.csv` with columns:
`attendance, study_hours, prev_marks, assignments, passed` (passed = 0/1).

## Live Demo
(add your Streamlit link here)
