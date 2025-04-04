from load_data import load_data
from feature_extraction import TF_idf, Word2vec
from clustering import kmeans_clustering, hierarchical_clustering
from evaluation import Silhoute_score
from visualization import plot_clusters , plot_dendrogram
from preprocessing import preprocessing
import joblib

print("-------------------------------------------")
print("Loading data...")
l = load_data()
print("-------------------------------------------")

print("Data preprocessing...")
data =[preprocessing(text) for text in l]
print("Preprocessing done.")
print("-------------------------------------------")

print("TF-IDF embedding...")
TF_IDF_embeddings , tfidf_vectorizer = TF_idf(data)
print("TF-IDF embedded Successfully.")
print("-------------------------------------------")

# print("Word2vec embedding...")
# Word2vec_embeddings = Word2vec(data)
# print("Word2vec embedded Successfully.")
# print("-------------------------------------------")

print("Kmeans clustering...")
Kmean_clusters , Clusters_labels = kmeans_clustering(TF_IDF_embeddings, n_clusters=3)
print("Kmeans clustering done.")
print("-------------------------------------------")


joblib.dump(Kmean_clusters, 'Models/kmeans_model.pkl')
joblib.dump(tfidf_vectorizer, 'Models/tfidf_vectorizer.pkl')
print("Kmeans model saved successfully!")
print("-------------------------------------------")


# print("Hierarchical clustering...")
# Agglomerative_clusters, Agglomerative_labels = hierarchical_clustering(Word2vec_embeddings, n_clusters=3)
# print("Hierarchical clustering done.")
# print("-------------------------------------------")

print("Calculating Silhouette score...")
Silhouette_score = Silhoute_score(TF_IDF_embeddings, Clusters_labels)
print(f"Silhouette score: {Silhouette_score}")
print("-------------------------------------------")

print("Plotting clusters...")
plot_clusters(TF_IDF_embeddings, Clusters_labels, title='Kmeans Clustering Visualization', save_path='results/Kmeans_clusters.png')
print("--------------------------------------------")

# print("Plotting dendrogram...")
# plot_dendrogram(Word2vec_embeddings, method='ward', title='Hierarchical Clustering Dendrogram', save_path='results/dendrogram.png')
print("All done!")