import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("rf_model.pkl")

st.title("Sepsis Risk Prediction App")

st.write("""
Enter the patient's clinical values to predict the risk of sepsis.
This model was trained on a cleaned dataset, so you will see warnings
if values are outside the range seen during training. This model was trained exclusively on patients with traumatic brain injury (TBI). 
Predictions may not generalize to other patient populations
""")

# --- User inputs ---
lactate = st.number_input("Lactate (mmol/L)", min_value=0.0, step=0.1, format="%.2f")
calcium = st.number_input("Calcium (mg/dL)", min_value=0.0, step=0.1, format="%.2f")
creatinine = st.number_input("Creatinine (mg/dL)", min_value=0.0, step=0.1, format="%.2f")
hemoglobin = st.number_input("Hemoglobin (g/dL)", min_value=0.0, step=0.1, format="%.2f")
platelets = st.number_input("Platelets (K/μL)", min_value=0.0, step=1.0)
white_blood_cells = st.number_input("White Blood Cells (K/μL)", min_value=0.0, step=0.1, format="%.2f")
age = st.number_input("Age", min_value=0, max_value=120, step=1)

invasive_ventilation = st.selectbox("Invasive Ventilation", ["Select...", "Yes", "No"])
niv_support_combined = st.selectbox("Non-Invasive Ventilation Support", ["Select...", "Yes", "No"])
urinary_catheter = st.selectbox("Urinary Catheter", ["Select...", "Yes", "No"])

# Validate categorical selections
valid_selection = all([
    invasive_ventilation in ["Yes", "No"],
    niv_support_combined in ["Yes", "No"],
    urinary_catheter in ["Yes", "No"]
])

# Check for missing or zero inputs
numeric_values = [lactate, calcium, creatinine, hemoglobin, platelets, white_blood_cells, age]

if st.button("Predict Sepsis Risk"):
    if not valid_selection or any(val == 0 for val in numeric_values):
        st.warning("⚠️ Please fill in all values. None can be zero or unselected.")
    else:
        # Convert categorical to binary
        inv = 1 if invasive_ventilation == "Yes" else 0
        niv = 1 if niv_support_combined == "Yes" else 0
        cath = 1 if urinary_catheter == "Yes" else 0

        # Create input array
        input_data = np.array([[lactate, inv, niv, cath,
                                white_blood_cells, platelets,
                                hemoglobin, creatinine, age, calcium]])

        # Prediction
        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0][1]

        st.subheader("Prediction Result")
        st.write(f"Prediction: **{'High risk of Sepsis' if prediction == 1 else 'Low risk of Sepsis'}**")
        st.write(f"Probability of Sepsis: **{proba:.2f}**")

        # Warn if values are out of training range
        if lactate > 10:
            st.warning("⚠️ Lactate is above the range seen during training (max 10 mmol/L).")
        if creatinine > 10:
            st.warning("⚠️ Creatinine is above the training max (10 mg/dL).")
        if calcium > 15:
            st.warning("⚠️ Calcium is above the training max (15 mg/dL).")
        if platelets < 18 or platelets > 970:
            st.warning("⚠️ Platelets are outside the range seen during training (18–970 K/μL).")
        if white_blood_cells < 0.9 or white_blood_cells > 155:
            st.warning("⚠️ White blood cells are outside training range (0.9–155 K/μL).")
        if hemoglobin < 20.6 or hemoglobin > 57.9:
            st.warning("⚠️ Hemoglobin is outside training range (20.6–57.9 g/dL).")
