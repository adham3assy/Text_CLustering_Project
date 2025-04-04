from sklearn.metrics import silhouette_score


def Silhoute_score(X, labels):
    """
    Calculate the silhouette score for a given set of data points and their corresponding labels.
    
    """
    return silhouette_score(X, labels)