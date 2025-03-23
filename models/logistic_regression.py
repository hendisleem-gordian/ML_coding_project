import math

class LogisticRegression:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.weights = []
        self.bias = 0

    def _sigmoid(self, z):
        '''just applying our lovely usual sigmoid function :)'''
        return 1 / (1 + math.exp(-z))

    def _predict_row(self, row):
        '''predict the probability of this row being class 1'''
        '''y = sigmoid(w·x + b)'''
        z = sum(w * x for w, x in zip(self.weights, row)) + self.bias
        return self._sigmoid(z)

    def fit(self, X, y):
        n_samples, n_features = len(X), len(X[0])
        self.weights = [0.0] * n_features
        self.bias = 0.0

        for epoch in range(self.epochs):
            for i in range(n_samples):
                y_pred = self._predict_row(X[i])
                error = y_pred - y[i]

                # updat the weights and the bias with gd
                for j in range(n_features):
                    self.weights[j] -= self.lr * error * X[i][j]
                self.bias -= self.lr * error

    def predict(self, X_data):
        '''Final convict'''
        return [1 if prob >= 0.5 else 0 for prob in [self._predict_row(row) for row in X_data]]

class MulticlassLogisticRegression:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.weights = []
        self.biases = []
        self.num_classes = 0

    def _softmax(self, z):
        max_z = max(z)
        # Hend: remember we subtract the max so we don't get e^ sth too large
        exp_z = [math.exp(i - max_z) for i in z]
        sum_exp = sum(exp_z)
        return [val / sum_exp for val in exp_z]

    def _predict_row(self, row):
        z = []
        for k in range(self.num_classes):
            score = sum(w * x for w, x in zip(self.weights[k], row)) + self.biases[k]
            z.append(score)
        return self._softmax(z)

    def fit(self, X, y):
        n_samples, n_features = len(X), len(X[0])
        self.num_classes = len(set(y))

        self.weights = [[0.0] * n_features for _ in range(self.num_classes)]
        self.biases = [0.0] * self.num_classes

        for epoch in range(self.epochs):
            for i in range(n_samples):
                y_true = y[i]
                probs = self._predict_row(X[i])

                for k in range(self.num_classes):
                    error = probs[k] - (1 if k == y_true else 0)
                    for j in range(n_features):
                        self.weights[k][j] -= self.lr * error * X[i][j]
                    self.biases[k] -= self.lr * error

    def predict(self, X_data):
        return [probs.index(max(probs)) for probs in [self._predict_row(row) for row in X_data]]
