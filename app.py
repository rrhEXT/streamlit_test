import streamlit as st
import joblib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# Load trained model and preprocessing tools
model = joblib.load("sdg_classifier.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Set page title and layout
st.set_page_config(page_title="Patent SDG Classifier", layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation")
tab = st.sidebar.radio("Go to", ["Home", "Analyze", "Insights"])

# --- Home Tab ---
if tab == "Home":
    st.title("Patent Classification Based on SDGs")
    st.write("This tool classifies patent abstracts into Sustainable Development Goals (SDGs).")

    # User input
    user_input = st.text_area("Enter patent abstract:", "")

    if st.button("Classify"):
        if user_input.strip():
            # Convert input text into TF-IDF features
            user_features = vectorizer.transform([user_input])
            
            # Predict SDG category
            prediction = model.predict(user_features)[0]
            predicted_sdg = label_encoder.inverse_transform([prediction])[0]
            
            st.success(f"Predicted SDG Category: **{predicted_sdg}**")
        else:
            st.warning("Please enter a patent abstract.")

# --- Analyze Tab ---
elif tab == "Analyze":
    st.title("Analyze Patent Trends")
    st.write("Upload a CSV file containing patent abstracts to classify multiple entries.")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        
        if "abstract" not in df.columns:
            st.error("CSV must contain an 'abstract' column.")
        else:
            # Process and predict for all abstracts
            df["features"] = vectorizer.transform(df["abstract"]).toarray().tolist()
            df["predicted_sdg"] = model.predict(vectorizer.transform(df["abstract"]))
            df["predicted_sdg"] = label_encoder.inverse_transform(df["predicted_sdg"])

            st.write("Classification Results:")
            st.dataframe(df[["abstract", "predicted_sdg"]])

            # Option to download results
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Results", csv, "classified_patents.csv", "text/csv")

# --- Insights Tab ---
elif tab == "Insights":
    st.title("Insights & Trends")
    st.write("Visualizations of SDG classification results, including **similarity analysis**.")

    uploaded_file = st.file_uploader("Upload CSV file for similarity analysis", type=["csv"], key="insights_upload")
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        if "abstract" not in df.columns:
            st.error("CSV must contain an 'abstract' column.")
        else:
            # Compute similarity matrix
            tfidf_matrix = vectorizer.transform(df["abstract"])
            similarity_matrix = cosine_similarity(tfidf_matrix)

            # Display similarity as a heatmap
            st.subheader("Patent Similarity Heatmap")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(similarity_matrix, cmap="coolwarm", xticklabels=False, yticklabels=False, ax=ax)
            st.pyplot(fig)

            # Display top similar patents
            st.subheader("Top Similar Patents")
            num_patents = min(len(df), 10)  # Show top 10 or all if fewer
            similarity_df = pd.DataFrame(similarity_matrix[:num_patents, :num_patents], 
                                         columns=[f"Patent {i+1}" for i in range(num_patents)], 
                                         index=[f"Patent {i+1}" for i in range(num_patents)])
            st.dataframe(similarity_df)

            st.info("Darker colors in the heatmap indicate higher similarity between patents.")

