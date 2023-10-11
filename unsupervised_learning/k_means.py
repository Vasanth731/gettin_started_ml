# clustering - k_means
import numpy as np
import matplotlib.pyplot as plt

# hyperparameters
np.random.seed(0)
data = np.random.randn(100, 2)
k = 3
centroids = data[np.random.choice(len(data), k, replace=False)]
max_iters = 100

def k_means(data, k, centroids, max_iters):
    for i in range(max_iters):
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([data[labels == j].mean(axis=0) for j in range(k)])
        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids
    return centroids

def main():
    global centroids
    centroids = k_means(data, k, centroids, max_iters)

    # plot
    plt.figure(figsize=(8, 6))
    labels = np.argmin(np.linalg.norm(data[:, np.newaxis] - centroids, axis=2), axis=1)
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='rainbow', marker='o', edgecolor='k', s=50, label='Data Points')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='x', s=100, label='Centroids')
    plt.title('K-means Clustering')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()

