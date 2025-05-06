# logistic regression for binary classification 
import numpy as np
from sklearn.datasets import make_blobs

class LogisticRegression:
    def  __init__(self, x, learning_rate=0.1, epochs=50):
        self.lr = learning_rate
        self.epochs = epochs
        self.m, self.n = x.shape
        # (m, n) = (training_examples, features)

    def train(self, x, y):
        self.weights = np.zeros((self.n, 1)) # col matrix
        self.bias = 0

        for i in range(self.epochs + 1):
            y_predict = self.sigmoid(np.dot(x, self.weights) + self.bias)

            loss = (-1 / self.m * np.sum(y * np.log(y_predict) + (1 - y) * np.log(1 - y_predict))) # cross entropy loss 

            dw = 1 / self.m * np.dot(x.T, (y_predict - y))
            db = 1 / self.m * np.sum(y_predict - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            if i % 5 == 0:
                print(f"Cost after iteration {i}: {loss}")

        return self.weights, self.bias


    def predict(self, x):
        y_predict = self.sigmoid(np.dot(x, self.weights) + self.bias)
        y_predict_labels = y_predict > 0.5
        return y_predict_labels # if y_predict > 0.5 returns True, else False

    # activation function
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    

if __name__ == "__main__":
    np.random.seed(1)
    x, y = make_blobs(n_samples=100, centers=2) 
    # x.shape = (100,2) - input
    # y.shape = (100,1) - target_output
    y = y[:, np.newaxis]

    logreg = LogisticRegression(x)
    w, b = logreg.train(x,y)
    y_predict = logreg.predict(x)

    print(f"Accuracy; {np.sum(y==y_predict)/x.shape[0]}")
