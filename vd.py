
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import matplotlib.pyplot as plt

def load_models():
    from sklearn.preprocessing import LabelEncoder
    df = pd.read_csv("students.csv")

    le_stress = LabelEncoder()
    df["StressLevelEncoded"] = le_stress.fit_transform(df["StressLevel"])

    for col in ["Gender", "Stream", "DietQuality"]:
        df[col] = LabelEncoder().fit_transform(df[col])

    features = ["Gender", "Age", "Stream", "Attendance (%)", "HoursStudiedPerDay", "HealthScore", "SleepHours", "DietQuality"]
    X = df[features]
    y_cgpa = df["CGPA"]
    y_stress = df["StressLevelEncoded"]

    rf_reg = RandomForestRegressor().fit(X, y_cgpa)
    rf_clf = RandomForestClassifier().fit(X, y_stress)

    return rf_reg, rf_clf, le_stress

# Load models
reg_model, clf_model, le_stress = load_models()

st.set_page_config(page_title="Student Predictor AI", layout="wide")

menu = st.sidebar.radio("Navigate", ["AI Guidelines", "Student Predictor"])

if menu == "AI Guidelines":
    st.title("Guidelines for AI Usage")
    st.image("AIbrain.jpeg", width=600)

    st.header("Good AI vs. Bad AI")
    st.write("### Good AI:")
    st.write("- Ethical, unbiased, and transparent.")
    st.write("- Helps in decision-making without discrimination.")
    st.write("- Enhances productivity and well-being.")
    st.write("- Provides accurate and fair predictions.")
    
    st.write("### Bad AI:")
    st.write("- Biased, unfair, or discriminatory.")
    st.write("- Misuses personal data.")
    st.write("- Generates misleading or harmful predictions.")
    st.write("- Lacks accountability and transparency.")
    
    st.write("By understanding these aspects, we can develop AI responsibly for student well-being.")
    st.image("robot.jpeg", width=600)

elif menu == "Student Predictor":
    st.title("Student Performance & Stress Predictor")
    st.markdown("Enter the student's information below to get predictions:")

    student_name = st.text_input("Student Name")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", min_value=17, max_value=22, value=19)
        stream = st.selectbox("Stream", ["Btech", "BBA", "BA"])
        diet = st.selectbox("Diet Quality", ["Poor", "Fair", "Good"])

    with col2:
        attendance = st.number_input("Attendance (%)", 10, 100, 85)
        hours_studied = st.number_input("Hours Studied Per Day", 0.5, 8.0, 4.0)
        health_score = st.number_input("Health Score (1–10)", 1, 10, 7)
        sleep_hours = st.number_input("Sleep Hours", 4.0, 10.0, 7.0)

    if st.button("Predict"):
        input_df = pd.DataFrame({
            "Gender": [0 if gender == "Male" else 1],
            "Age": [age],
            "Stream": [0 if stream == "Science" else 1 if stream == "Commerce" else 2],
            "Attendance (%)": [attendance],
            "HoursStudiedPerDay": [hours_studied],
            "HealthScore": [health_score],
            "SleepHours": [sleep_hours],
            "DietQuality": [0 if diet == "Poor" else 1 if diet == "Fair" else 2]
        })

        predicted_cgpa = reg_model.predict(input_df)[0]
        predicted_stress = le_stress.inverse_transform(clf_model.predict(input_df))[0]

        st.success(f"Predicted CGPA for **{student_name}**: **{predicted_cgpa:.2f}**")
        st.info(f"Predicted Stress Level for **{student_name}**: **{predicted_stress}**")

        # Visualization
        future_path = "High" if predicted_cgpa >= 8.0 and predicted_stress == "Low" else \
                      "Moderate" if predicted_cgpa >= 6.0 else "Needs Support"

        st.subheader("Future Outlook")
        fig, ax = plt.subplots()
        categories = ["Academic", "Health", "Stress"]
        values = [
            predicted_cgpa,
            health_score,
            10 if predicted_stress == "Low" else 5 if predicted_stress == "Medium" else 2
        ]

        ax.bar(categories, values, color=['skyblue', 'lightgreen', 'salmon'])
        ax.set_ylim(0, 10)
        st.pyplot(fig)

        st.markdown(f"**Predicted Future Path: _{future_path}_**")

        # Smart Suggestions
        st.subheader("Smart Suggestions")
        if sleep_hours < 7:
            st.markdown("-Try to improve sleep by 1 hour for lower stress.")
        if hours_studied < 5:
            st.markdown("-Studying 1 more hour daily can improve your CGPA.")
        if diet == "Poor":
            st.markdown("-Consider improving diet for better energy and stress levels.")
        if attendance < 75:
            st.markdown("-Higher attendance could enhance academic performance.")

    st.markdown("---")

