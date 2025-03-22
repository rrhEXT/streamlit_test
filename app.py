import streamlit as st
import joblib
import numpy as np

# Load the trained SDG classifier model
@st.cache_resource
def load_model():
    return joblib.load("sdg_classifier.pkl")  # Make sure this model exists

model = load_model()

# Streamlit UI
st.title("🔍 AI-Powered SDG Patent Classifier")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Upload Patent", "Analyze", "Insights"])

if page == "Upload Patent":
    st.subheader("Upload a Patent Abstract")
    user_input = st.text_area("Enter a patent abstract:", "")

    if st.button("Classify"):
        if user_input:
            # Predict SDG
            prediction = model.predict([user_input])[0]
            st.success(f"🟢 Predicted SDG: **SDG {prediction}**")
        else:
            st.warning("⚠️ Please enter a patent abstract.")

elif page == "Analyze":
    st.subheader("AI Confidence Score")
    st.bar_chart(np.random.rand(17))  # Placeholder for SDG confidence scores

elif page == "Insights":
    st.subheader("SDG Distribution")
    st.pie_chart(np.random.rand(17))  # Placeholder for SDG category distribution
