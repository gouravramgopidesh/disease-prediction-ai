import streamlit as st
import pickle
import pandas as pd
import os


# Load model
model = pickle.load(open(os.path.join("model", "model.pkl"), "rb"))

# Load dataset
data = pd.read_csv("data/dataset.csv")

# Symptoms list
symptoms = [col for col in data.columns if col != "disease"]

# Disease info
disease_info = {
    "Flu": "Viral infection with fever and body pain.",
    "Cold": "Mild infection affecting nose and throat.",
    "Migraine": "Severe headache with nausea.",
    "Infection": "General infection causing fever.",
    "Allergy": "Reaction causing cough and runny nose."
}

# UI
st.title("🧑‍⚕️ AI Disease Prediction System")

selected = st.multiselect("Select Symptoms", symptoms)

# Input vector
# Ensure same features as training
model_features = model.feature_names_in_

# Create input with all features
input_data = {feature: 0 for feature in model_features}

# Fill selected symptoms
for s in selected:
    if s in input_data:
        input_data[s] = 1

# Convert to DataFrame in correct order
import pandas as pd
input_df = pd.DataFrame([input_data])

# Ensure column order matches model
input_df = input_df[model_features]

# Predict
prediction = model.predict(input_df)[0]

# Predict
if st.button("Predict Disease"):

    if not selected:
        st.warning("Select symptoms first")
    else:
        df = pd.DataFrame([input_data])

        prediction = model.predict(df)[0]
        probs = model.predict_proba(df)[0]

        st.success(f"Most Likely: {prediction}")
        st.write("Description:", disease_info.get(prediction, "N/A"))

        st.subheader("Confidence")
        st.progress(float(max(probs)))

        st.subheader("Top Predictions")

        classes = model.classes_
        top = probs.argsort()[-3:][::-1]

        for i in top:
            st.write(f"{classes[i]} → {probs[i]:.2f}")

# Disclaimer
st.warning("⚠️ Not a medical diagnosis. Consult a doctor.")