# Text Clustering Project: People Wikipedia Dataset

## Overview

This project applies **unsupervised learning** techniques to cluster biographical articles from the **People Wikipedia Dataset**. The goal is to uncover relationships between individuals based on textual similarities in their biographies, such as professional background, historical significance, or shared attributes. The ultimate aim is to group individuals into meaningful clusters that reveal hidden patterns, insights, and connections.

---

## Dataset: People Wikipedia

### Description

The **People Wikipedia Dataset** contains structured biographical articles of notable individuals extracted from Wikipedia. Each entry includes biographical information such as profession, achievements, and historical context. This dataset serves as a foundation for exploring how people can be grouped based on shared attributes or life events.

### Features

- **URI**: A unique identifier for each person’s Wikipedia page.
- **Name**: The full name of the individual.
- **Text**: Extracted biography containing detailed information about their profession, achievements, and historical context.

### Use Cases

- Group individuals based on shared professional backgrounds, life events, or common themes.
- Identify patterns in various domains like professions, historical significance, or geography.
- Provide insights into the relationships between individuals, helping researchers understand how people are connected.

---

## Methodology

### 1. Data Preprocessing 🪹

Before applying clustering algorithms, the raw data needs to be cleaned and processed:

- **Text Cleaning**: Remove punctuation, special characters, and stop words.
- **Tokenization**: Split text into words or phrases (tokens) to prepare for feature extraction.
- **Lemmatization**: Normalize words to their base form (e.g., "running" → "run").



### 2. Feature Extraction 📃

To convert the text data into numerical features, several methods are applied:

- **TF-IDF Vectorization**: Converts text into numerical features based on Term Frequency-Inverse Document Frequency (TF-IDF). This method highlights important words in each biography while minimizing common words that don't add much value.
- **Word Embeddings (Optional)**: Techniques such as **Word2Vec** or **GloVe** are used to create semantic representations of words, capturing contextual meaning and relationships between words across documents.

### 3. Clustering Algorithms 🧠

We apply various unsupervised clustering algorithms to group the biographies:

- **K-Means Clustering**: Partitions individuals into k clusters based on feature similarity. Optimal k is determined using methods like the **elbow method** or **silhouette analysis**.
- **Hierarchical Clustering**: Builds a tree-like structure of nested clusters (dendrograms), allowing for hierarchical grouping.

### 4. Evaluation Metrics 📊

To assess the quality of clustering results:

- **Silhouette Score**: Measures how similar an individual is to its own cluster compared to other clusters. Higher scores indicate better-defined clusters.

### 5. Visualization Techniques 📉

We visualize clustering results to aid understanding and interpretability:

- **Principal Component Analysis (PCA)**: Reduces the dimensionality of the feature space for easy visualization of clusters.
- **Dendrograms**: For hierarchical clustering, dendrograms visually represent how clusters are merged.

---

## 📁 Project Structure

```
people-wikipedia-clustering/
├── 📦 Models/                        # Trained clustering models
│   ├── kmeans_model.pkl             # Saved K-Means model
│   └── hierarchical_model.pkl       # Saved Hierarchical model
│
├── 🧑‍💻 notebooks/                   # Jupyter notebooks for analysis
│   ├── 01_data_exploration.ipynb    # Data loading and EDA
│   └── 02_clustering_analysis.ipynb # Clustering and visualization
│
├── 📊 results/                      # Clustering results and plots
│   ├── clusters.csv                 # Cluster labels per biography
│   └── cluster_visualizations/     # PCA, t-SNE, dendrogram images
│
├── 📝 src/                          # Source code for pipeline
│   ├── preprocessing.py            # Text cleaning and lemmatization
│   ├── feature_extraction.py       # TF-IDF & embedding generation
│   ├── clustering.py               # K-Means & Hierarchical logic
│   ├── evaluation.py               # Metrics: silhouette, ARI, etc.
│   ├── visualization.py            # Dimensionality reduction, plots
│   └── main.py                     # Run full pipeline end-to-end
│
├── 📁 requirements.txt             # Python dependencies
├── 📄 README.md                    # Project documentation (this file)
└── 🚀 app.py                       # (Optional) Web interface or API
```

---

## Technologies Used ⚙️

- **Programming Language**: Python
- **Libraries**:
  - **Data Handling**: pandas, NumPy
  - **Text Processing**: NLTK , regex
  - **Machine Learning**: scikit-learn, gensim
  - **Visualization**: matplotlib, seaborn

---


