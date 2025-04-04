from sklearn.cluster import KMeans, AgglomerativeClustering


def kmeans_clustering(X, n_clusters=3):
    """
    Performs KMeans clustering on input data X.
    Returns the trained KMeans model and cluster labels.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(X)
    return kmeans, kmeans.labels_

def hierarchical_clustering(X, n_clusters=3):
    """
    Performs Agglomerative (Hierarchical) Clustering on input data X.
    Returns the trained AgglomerativeClustering model and cluster labels.
    """
    agglomerative = AgglomerativeClustering(n_clusters=n_clusters)
    agglomerative.fit(X)
    return agglomerative, agglomerative.labels_