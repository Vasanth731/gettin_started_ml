# k_nearest_neighbors
import numpy as np
import matplotlib.pyplot as plt

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = np.bincount(k_nearest_labels).argmax()
        return most_common

def main():
    np.random.seed(0)
    X_train = np.random.rand(100, 2) # input with two features
    y_train = (X_train[:, 0] + X_train[:, 1] > 1).astype(int) # output with binary labels
    
    X_test = np.random.rand(10, 2)

    knn = KNN(k=3)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    # accuracy
    correct_predictions = np.sum(y_pred == y_train[:10]) 
    total_predictions = len(y_pred)
    accuracy = correct_predictions / total_predictions
    print(f"Accuracy: {accuracy * 100:.2f}%")
    
    # plot
    plt.figure(figsize=(8, 6))
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', marker='o', s=50, label='Training Data')
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='coolwarm', marker='x', s=100, label='Test Predictions')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.title('KNN Plot')
    plt.show()

if __name__ == "__main__":
    main()
