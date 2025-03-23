import math, os

import numpy as np


class LogisticRegression:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.weights = []
        self.bias = 0
        self.training_accuracy = [] # Hend: this's to keep track of the training accuracy (boring not seeing a thing)
        self.training_loss = [] # same here

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

        os.makedirs("history", exist_ok=True)
        log_file = open("history/logistic_regression_training_log.txt", "a")
        log_file.write("\\n--- New Training Session ---\\n")
        log_file.write("-----------------------------------------------------------------\n")

        for epoch in range(self.epochs):
            for i in range(n_samples):
                y_pred = self._predict_row(X[i])
                error = y_pred - y[i]

                # updat the weights and the bias with gd
                for j in range(n_features):
                    self.weights[j] -= self.lr * error * X[i][j]
                self.bias -= self.lr * error

            
            # Accuracy tracking
            preds = self.predict(X)
            acc = sum(1 for p, y_true in zip(preds, y) if p == y_true) / n_samples
            self.training_accuracy.append(acc)

            # Loss calculation (cross-entropy)
            total_loss = 0.0
            for xi, yi in zip(X, y):
                probs = self._predict_row(xi)
                total_loss -= yi * math.log(probs + 1e-15) + (1 - yi) * math.log(1 - probs + 1e-15)
            avg_loss = total_loss / n_samples
            self.training_loss.append(avg_loss)

            # every 100 epochs
            if (epoch + 1) % 100 == 0 or epoch == self.epochs - 1:
                log_msg = f"Epoch {epoch + 1}/{self.epochs} - Accuracy: {acc:.4f} - Loss: {avg_loss:.4f}"
                print(log_msg)
                log_file.write(log_msg + "\n")
        

        log_file.write("-----------------------------------------------------------------\n")
        log_file.close()

    def predict(self, X_data):
        '''Final convict'''
        return [1 if prob >= 0.5 else 0 for prob in [self._predict_row(row) for row in X_data]]




class OldMulticlassLogisticRegression:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.weights = []
        self.biases = []
        self.num_classes = 0
        self.training_accuracy = []
        self.training_loss = []

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


        os.makedirs("history", exist_ok=True)
        log_file = open("history/multiclass_logistic_regression_training_log.txt", "a")
        log_file.write("\\n--- New Training Session ---\\n")
        log_file.write("-----------------------------------------------------------------\n")


        for epoch in range(self.epochs):
            for i in range(n_samples):
                y_true = y[i]
                probs = self._predict_row(X[i])

                for k in range(self.num_classes):
                    error = probs[k] - (1 if k == y_true else 0)
                    for j in range(n_features):
                        self.weights[k][j] -= self.lr * error * X[i][j]
                    self.biases[k] -= self.lr * error

            
            # Accuracy tracking
            preds = self.predict(X)
            acc = sum(1 for p, y_true in zip(preds, y) if p == y_true) / n_samples
            self.training_accuracy.append(acc)

            # Loss calculation (cross-entropy)
            total_loss = 0.0
            for xi, yi in zip(X, y):
                probs = self._predict_row(xi)
                total_loss -= math.log(probs[yi] + 1e-15)  # avoid log(0)
            avg_loss = total_loss / n_samples
            self.training_loss.append(avg_loss)

            # every 100 epochs
            if (epoch + 1) % 100 == 0 or epoch == self.epochs - 1:
                log_msg = f"Epoch {epoch + 1}/{self.epochs} - Accuracy: {acc:.4f} - Loss: {avg_loss:.4f}"
                print(log_msg)
                log_file.write(log_msg + "\n")
        

        log_file.write("-----------------------------------------------------------------\n")
        log_file.close()

    def predict(self, X_data):
        return [probs.index(max(probs)) for probs in [self._predict_row(row) for row in X_data]]



class MulticlassLogisticRegression:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.biases = None
        self.num_classes = 0
        self.training_accuracy = []
        self.training_loss = []

    def _softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        n_samples, n_features = X.shape
        self.num_classes = len(np.unique(y))

        # One-hot encode labels coz I want the outputs of softmax [p1, p2, p3, ...] to match the actual true label which is currently a number. and that's why we use the one-hot so it can be like [0, 0, 1, 0, ...]
        y_one_hot = np.zeros((n_samples, self.num_classes))
        y_one_hot[np.arange(n_samples), y] = 1

        self.weights = np.zeros((n_features, self.num_classes))
        self.biases = np.zeros((1, self.num_classes))

        os.makedirs("history", exist_ok=True)
        with open("history/multiclass_logistic_regression_training_log.txt", "a") as log_file:
            log_file.write("\n--- New Training Session ---\n")

            for epoch in range(self.epochs):
                logits = np.dot(X, self.weights) + self.biases
                probs = self._softmax(logits)

                # Loss calculation (cross-entropy)
                loss = -np.mean(np.sum(y_one_hot * np.log(probs + 1e-15), axis=1))
                self.training_loss.append(loss)

                predictions = np.argmax(probs, axis=1)
                acc = np.mean(predictions == y)
                self.training_accuracy.append(acc)

                error = probs - y_one_hot
                dw = np.dot(X.T, error) / n_samples
                db = np.sum(error, axis=0, keepdims=True) / n_samples

                self.weights -= self.lr * dw
                self.biases -= self.lr * db

                if (epoch + 1) % 100 == 0 or epoch == self.epochs - 1:
                    log_msg = f"Epoch {epoch + 1}/{self.epochs} - Accuracy: {acc:.4f} - Loss: {loss:.4f}"
                    print(log_msg)
                    log_file.write(log_msg + "\n")

    def predict_proba(self, X):
        X = np.array(X)
        logits = np.dot(X, self.weights) + self.biases
        return self._softmax(logits)

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1).tolist()
