import numpy as np

# y = {-1, 1} in this context
class Custom_Perceptron:
    
    def __init__(self, dimension):
        self.dimension = dimension
        self.weight = np.random.normal(size=dimension)
        self.b = 0
    
    def score(self, x, y):
        return (x.dot(self.weight) + self.b) * (1 if y else -1)
    
    def predict(self, x):
        return x.dot(self.weight) + self.b > 0
    
    def fit_one_datapoint(self, x, y):
        pred = self.predict(x)
        pred_sign = 1 if pred else -1
        label_sign = 1 if y else -1
        self.weight += (label_sign - pred_sign) / 2 * x
        self.b += (label_sign - pred_sign) / 2

    def fit_iter(self, X, y):
        data_count = len(y)
        for data_idx in range(data_count):
            self.fit_one_datapoint(X[data_idx], y[data_idx])

if __name__ == '__main__':
    # creating data points
    dimension, data_count, iter_count = 5, 100, 100
    weight = np.random.normal(size=dimension)
    X = np.random.normal(size=(data_count, dimension))
    labels = X.dot(weight) - 1 > 0

    # perceptron
    p = Custom_Perceptron(dimension)
    for _ in range(iter_count):
        p.fit_iter(X, labels)
    assert(not np.logical_xor(p.predict(X), labels).any())
    print('Perceptron converges for linearly separable cases')
