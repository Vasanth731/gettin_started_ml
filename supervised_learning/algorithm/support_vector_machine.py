# the SVM is taken from aladdinpersson and the credit goes to him, just the wrapper for multi-class is new here

# imports
import numpy as np
import cvxopt
import matplotlib.pyplot as plt

# dataset
def create_dataset(N, D=2, K=3):  # N = Samples per class, D = Dimensions, K = Classes
    X = np.zeros((N * K, D))
    y = np.zeros(N * K)

    for j in range(K):
        ix = range(N * j, N * (j + 1))
        r = np.linspace(0.0, 1, N)  # radius
        t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2  # theta
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j

    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Spectral)
    plt.title("Generated Dataset")
    plt.show()

    return X, y

# prediction
def plot_contour(X, y, svm):
    h = 0.01
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    points = np.c_[xx.ravel(), yy.ravel()]

    Z = svm.predict(points)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.title("SVM Decision Boundaries")
    plt.show()

# kernels
def linear(x, z):
    return np.dot(x, z.T)

def polynomial(x, z, p=5):
    return (1 + np.dot(x, z.T)) ** p

def gaussian(x, z, sigma=0.1):
    return np.exp(-np.linalg.norm(x - z, axis=1) ** 2 / (2 * sigma ** 2))

# binary SVM
class SVM:
    def __init__(self, kernel=gaussian, C=1):
        self.kernel = kernel
        self.C = C

    def fit(self, X, y):
        self.y = y
        self.X = X
        m, n = X.shape

        # Kernel matrix
        self.K = np.zeros((m, m))
        for i in range(m):
            self.K[i, :] = self.kernel(X[i, np.newaxis], self.X)

        P = cvxopt.matrix(np.outer(y, y) * self.K)
        q = cvxopt.matrix(-np.ones((m, 1)))
        G = cvxopt.matrix(np.vstack((-np.eye(m), np.eye(m))))
        h = cvxopt.matrix(np.hstack((np.zeros(m), np.ones(m) * self.C)))
        A = cvxopt.matrix(y, (1, m), "d")
        b = cvxopt.matrix(np.zeros(1))
        cvxopt.solvers.options["show_progress"] = False

        sol = cvxopt.solvers.qp(P, q, G, h, A, b)
        self.alphas = np.array(sol["x"])

    def predict_decision(self, X):
        y_decision = np.zeros((X.shape[0]))
        sv = self.get_parameters(self.alphas)

        for i in range(X.shape[0]):
            y_decision[i] = np.sum(
                self.alphas[sv]
                * self.y[sv, np.newaxis]
                * self.kernel(X[i], self.X[sv])[:, np.newaxis]
            )

        return y_decision + self.b

    def predict(self, X):
        return np.sign(self.predict_decision(X))

    def get_parameters(self, alphas):
        threshold = 1e-5
        sv = ((alphas > threshold) * (alphas < self.C)).flatten()
        self.w = np.dot(self.X[sv].T, alphas[sv] * self.y[sv, np.newaxis])
        self.b = np.mean(
            self.y[sv, np.newaxis]
            - self.alphas[sv]
            * self.y[sv, np.newaxis]
            * self.K[sv][:, sv][:, np.newaxis]
        )
        return sv

# multi-class wrapper
class MultiClassSVM:
    def __init__(self, kernel=gaussian, C=1):
        self.kernel = kernel
        self.C = C
        self.models = []

    def fit(self, X, y):
        self.classes = np.unique(y).astype(int)
        self.models = []

        for cls in self.classes:
            y_binary = np.where(y == cls, 1, -1)
            model = SVM(kernel=self.kernel, C=self.C)
            model.fit(X, y_binary)
            self.models.append(model)

    def predict(self, X):
        # decision values from each binary classifier
        decisions = np.array([model.predict_decision(X) for model in self.models])
        return self.classes[np.argmax(decisions, axis=0)]

if __name__ == "__main__":
    np.random.seed(1)
    X, y = create_dataset(N=50, K=4)

    svm = MultiClassSVM(kernel=gaussian)
    svm.fit(X, y)
    y_pred = svm.predict(X)

    plot_contour(X, y, svm)
    print(f"Accuracy: {np.mean(y == y_pred):.2f}")
