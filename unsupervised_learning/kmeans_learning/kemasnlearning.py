import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt




class KMeans():
    def __init__(self,k = 2, max_iterations=500):
        self.k = k
        self.max_iterations = max_iterations


    @staticmethod
    def euclidean_distance(x1,x2):
        return np.linalg.norm(x1-x2)


    def get_random_centroid(self,x):
        n_samples, n_features = np.shape(x)
        centroids = np.zeros((self.k,n_features))

        for i in range(self.k):
            centroid = x[np.random.choice(range(n_samples))]
            centroids[i]= centroid
        return centroids

    def closest_centroid(self,sample,centroids):
        closest_i = 0
        closest_dist = float('inf')
        for i , centroid in enumerate(centroids):
            distance = self.euclidean_distance(sample,centroid)
            if distance < closest_dist:
                closest_i = i
                closest_dist = distance
        return closest_i


    def create_clusters(self, centroids, x):
        clusters = [[] for _ in range(self.k)]
        for sample_i,sample in enumerate(x):
            centroid_i = self.closest_centroid(sample,centroids)
            clusters[centroid_i].append(sample_i)
        return clusters

    def calculate_centrioids(self,clusters,x):
        n_features = np.shape(x)[1]
        centroids = np.zeros((self.k,n_features))
        for i , cluster in enumerate(clusters):
            centroid = np.mean(x[cluster] , axis= 0)
            centroids[i] = centroid
        return centroids

    def get_clusters_label(self,clusters,x):
        y_pred = np.zeros((np.shape(x)[0]))
        for cluster_i, cluster in enumerate(clusters):
            for sample_i in cluster:
                y_pred[sample_i] =  cluster_i
        return y_pred

    def predict(self,x):
        centroids = self.get_random_centroid(x)

        for _ in range(self.k):
            clusters =self.create_clusters(centroids,x)

            prev_centroids = centroids

            centroids = self.calculate_centrioids(clusters,x)

            diff = centroids - prev_centroids

            if not diff.any():
                break

        return self.get_clusters_label(clusters,x)


# Generate synthetic data
X, _ = make_blobs(n_samples=300, centers=2, random_state=42)

# Scatter plot of the dataset
plt.scatter(X[:, 0], X[:, 1], s=50, cmap='viridis')
plt.title("Generated Data")
plt.show()


# Import your KMeans class (assuming it's in the same script or module)
kmeans = KMeans()

# Predict clusters
y_pred = kmeans.predict(X)

# Plot the clustered data
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', s=50)
plt.title("K-Means Clustering Result")
plt.show()
