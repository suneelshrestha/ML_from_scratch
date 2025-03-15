from k_means import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

kmeans = KMeans()

# Generate synthetic data
X, _ = make_blobs(n_samples=300, centers=3, random_state=42)

# Scatter plot of the dataset
plt.scatter(X[:, 0], X[:, 1])
plt.title("Generated Data")
plt.show()


# Import your KMeans class (assuming it's in the same script or module)
kmeans = KMeans(k=3, max_iterations=500)

# Predict clusters
y_pred = kmeans.predict(X)

# Plot the clustered data
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title("K-Means Clustering Result")
plt.show()
