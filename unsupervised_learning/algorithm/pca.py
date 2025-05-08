import numpy as np
import matplotlib.pyplot as plt # for 2D plot
from mpl_toolkits.mplot3d import Axes3D  # for 3D plot

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.explained_variance = None
    
    def fit(self, X):
        # mean normalization
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # covariance matrix
        cov_matrix = np.cov(X_centered, rowvar=False)

        # eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        # sorting the eigenvalues and corresponding eigenvectors
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # top n_components
        self.components = eigenvectors[:, :self.n_components]
        print(f"components_shape : {self.components.shape}")

        # Calculate explained variance
        total_variance = np.sum(eigenvalues)
        self.explained_variance = eigenvalues[:self.n_components] / total_variance

    def transform(self, X):
        # projecting the data onto the selected components
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)

    def plot_explained_variance(self):
        labels = [f'PCA{i+1}' for i in range(self.n_components)]
        plt.figure(figsize=(8, 6))
        plt.bar(range(1, self.n_components + 1), self.explained_variance, alpha=0.7, align='center', color='blue', tick_label=labels)
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.title('Explained Variance by Principal Components')
        plt.show()

    def plot2D(self):
        plt.figure(figsize=(8, 6))
        plt.scatter(X_transformed[:, 0], X_transformed[:, 1], alpha=0.7, c='teal')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('Data Projected onto First 2 Principal Components')
        plt.grid(True)
        plt.show()
    
    def plot3D(self):
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X_transformed[:, 0], X_transformed[:, 1], X_transformed[:, 2], alpha=0.7, c='teal')
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_zlabel('Principal Component 3')
        ax.set_title('Data Projected onto First 3 Principal Components')
        plt.show()

if __name__ == "__main__":

    # input
    np.random.seed(42)
    low_dim_data = np.random.randn(100, 3)
    projection_matrix = np.random.randn(3, 10)
    high_dim_data = np.dot(low_dim_data, projection_matrix)
    noise = np.random.normal(loc=0, scale=0.5, size=(100, 10))
    data_with_noise = high_dim_data + noise 
    X = data_with_noise
    print(f"input_shape : {X.shape}")

    # pca
    pca = PCA(n_components=2) # n_components is the dimension to what it has to be reduced 
    pca.fit(X)
    X_transformed = pca.transform(X)
    print(f"transformed_X_shape : {X_transformed.shape}")
    pca.plot_explained_variance()
    # so explained variance tells how much percentege of the original variance is captured by the principal component
    print("Explained Variance:\n", pca.explained_variance)

    # plot
    if pca.n_components == 2 :
        pca.plot2D()
    elif pca.n_components == 3 :
        pca.plot3D()

