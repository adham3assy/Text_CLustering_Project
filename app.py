
import streamlit as st
import joblib
import numpy as np
from src.preprocessing import preprocessing
from src.feature_extraction import TF_idf
from src.clustering import kmeans_clustering
import matplotlib.pyplot as plt
from src.visualization import plot_clusters
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Load the trained model and vectorizer
kmeans_model = joblib.load('Models/kmeans_model.pkl')
tfidf_vectorizer = joblib.load('Models/tfidf_vectorizer.pkl')

st.title("🔍 Text Clustering Web App")

# **📌 Option 1: Single Text Prediction**
st.header("📝 Predict Cluster for a Single Text")
user_text = st.text_area("Enter a text:", "")

if user_text:
    cleaned_text = preprocessing(user_text)
    text_vector = tfidf_vectorizer.transform([cleaned_text])
    predicted_cluster = kmeans_model.predict(text_vector)[0]

    st.write(f"📌 **Predicted Cluster:** {predicted_cluster}")

# **📌 Option 2: Upload a Dataset for Batch Predictions**
st.header("📂 Predict Clusters for a Dataset")
uploaded_file = st.file_uploader("Upload a CSV file with a 'text' column", type=["csv"])

if uploaded_file is not None:
    # **📌 Step 1: Read the uploaded file**
    df = pd.read_csv(uploaded_file)
    st.write("📌 **Uploaded Dataset Preview:**", df.head())

    # **📌 Step 2: Preprocess the text**
    df["Cleaned_Text"] = df["text"].apply(preprocessing)

    # **📌 Step 3: Extract features using TF-IDF**
    text_features = tfidf_vectorizer.transform(df["Cleaned_Text"])

    # **📌 Step 4: Predict clusters**
    df["Cluster"] = kmeans_model.predict(text_features)

    # Show the results
    st.write("📌 **Clustered Dataset Preview:**", df.head())

    # **📌 Step 5: Calculate Silhouette Score**
    if len(set(df["Cluster"])) > 1:  # Ensure there are at least 2 clusters
        silhouette_avg = silhouette_score(text_features, df["Cluster"])
        st.write(f"📊 **Silhouette Score:** {silhouette_avg:.3f}")
    else:
        st.write("⚠️ Not enough clusters to compute Silhouette Score.")

    # **📌 Step 6: Visualization of Clusters**
    if st.checkbox("📈 Show Cluster Visualization"):
        # Reduce dimensions using PCA for plotting
        num_samples, num_features = text_features.shape
        if num_features > 1:  # Apply PCA only if features > 1
            X_embedded = PCA(n_components=2).fit_transform(text_features.toarray())

            # Convert PCA results to DataFrame
            vis_df = pd.DataFrame(X_embedded, columns=["PCA1", "PCA2"])
            vis_df["Cluster"] = df["Cluster"]

            # Plot clusters
            plt.figure(figsize=(8, 5))
            sns.scatterplot(x="PCA1", y="PCA2", hue="Cluster", data=vis_df, palette="viridis")
            plt.title("KMeans Clustering Visualization")
            st.pyplot(plt)
        else:
            st.write("⚠️ Not enough features for PCA visualization.")

    # **📌 Step 7: Download the results**
    def convert_df_to_csv(dataframe):
        return dataframe.to_csv(index=False).encode("utf-8")

    csv_data = convert_df_to_csv(df)

    st.download_button(
        label="📥 Download Clustered Data",
        data=csv_data,
        file_name="clustered_data.csv",
        mime="text/csv",
    )

