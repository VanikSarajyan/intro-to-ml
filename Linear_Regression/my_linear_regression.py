import numpy as np

class MyLinearRegression:
    def __init__(self, learning_rate = 0.001):
        self.learning_rate = learning_rate
        self.W = None
        self.b = None
    
    def fit(self, X, y, epochs = 1000):
        n_samples, n_features = X.shape
        self.W = np.zeros(n_features)
        self.b = 0

        #Gradient decent
        for _ in range(epochs):
            y_pred = np.dot(X, self.W) + self.b

            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)

            self.W -= self.learning_rate * dw
            self.b -= self.learning_rate * db
        
        return self

    def predict(self, X):
        return np.dot(X, self.W) + self.b
