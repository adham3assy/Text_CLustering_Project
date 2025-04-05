# Text Clustering Project (People Wikipedia Dataset)

## Overview  

This project applies **unsupervised learning techniques** to cluster biographical articles from the **People Wikipedia Dataset**. The goal is to uncover relationships between individuals based on textual similarities in their biographies, such as professional background, historical significance, or shared attributes.  

## Dataset: People Wikipedia  

### Description  
The **People Wikipedia Dataset** consists of structured biographical articles of notable individuals extracted from Wikipedia. The dataset provides an opportunity to analyze how people can be grouped based on the content of their biographies.  

### Features  
- **URI**: A unique identifier for each person’s Wikipedia page.  
- **Name**: The full name of the individual.  
- **Text**: Extracted biography from Wikipedia, containing information about their profession, achievements, and historical context.  

### Use Cases  
- Cluster individuals based on textual similarities.  
- Identify common patterns in professions, historical relevance, or fields of work.  
- Provide meaningful insights into biographical relationships.  

## Methodology  

### 1. Data Preprocessing  
- **Text Cleaning**: Removing punctuation, special characters, and stop words.  
- **Tokenization**: Splitting text into words or phrases.  
- **Lemmatization**: Normalizing words to their base form.  

### 2. Feature Extraction  
- **TF-IDF Vectorization**: Converts text into numerical features based on term frequency and inverse document frequency.  
- **Word Embeddings (Optional)**: Techniques like **Word2Vec** or **GloVe** for better semantic representation.  

### 3. Clustering Algorithms  
- **K-Means Clustering**: Groups biographies based on feature similarity.  
- **Hierarchical Clustering**: Creates a tree-based structure of nested clusters.    

### 4. Evaluation Metrics  
- **Silhouette Score**: Measures cluster cohesion and separation.  
    
### 5. Visualization Techniques  
- **PCA**: Reducing dimensionality for cluster visualization.  
- **Dendrograms**: For hierarchical clustering visualization.  

## Project Structure  

├── Models/ # Trained clustering models
├── notebooks/ # Jupyter notebooks for data exploration
├── results/ # Cluster results and visualizations
├── src/
│ ├── preprocessing.py # Text cleaning and preprocessing
│ ├── feature_extraction.py # TF-IDF and embeddings
│ ├── clustering.py # Clustering algorithm implementations
│ ├── evaluation.py # Metrics calculation
│ ├── visualization.py # Graphs and plots
│ ├── main.py # Pipeline execution
├── requirements.txt # Required Python libraries
├── README.md # Project documentation
└── app.py # Main script for deployment

## Technologies Used  

- **Programming Language**: Python  
- **Libraries**:  
  - **Data Handling**: pandas, NumPy  
  - **Text Processing**: NLTK 
  - **Machine Learning**: scikit-learn, gensim  
  - **Visualization**: matplotlib, seaborn 
