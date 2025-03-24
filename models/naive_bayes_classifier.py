import math
from collections import defaultdict, Counter


class NaiveBayesClassifier:
    '''The well-known smart voting system xD'''
    def __init__(self):
        self.class_priors = {}
        self.feature_probs = {}  

    def fit(self, x, y):
        n_samples = len(x)
        n_features = len(x[0])
        class_counts = Counter(y)
        self.class_priors = {c: count / n_samples for c, count in class_counts.items()}

        
        self.feature_probs = {c: [defaultdict(int) for _ in range(n_features)] for c in class_counts}

        for xi, yi in zip(x, y):  # features probabilities for each class
            for idx, val in enumerate(xi):
                self.feature_probs[yi][idx][val] += 1

        # Convert counts to probabilities with Laplace smoothing
        for c in self.feature_probs:
            for idx in range(n_features):
                total = sum(self.feature_probs[c][idx].values())
                unique_vals = len(self.feature_probs[c][idx])
                for val in self.feature_probs[c][idx]:
                    self.feature_probs[c][idx][val] = (self.feature_probs[c][idx][val] + 1) / (total + unique_vals)

    def predict(self, x):
        predictions = []
        for row in x:
            class_scores = {}
            for c in self.class_priors:
                log_prob = math.log(self.class_priors[c])
                for idx, val in enumerate(row):
                    prob = self.feature_probs[c][idx].get(val, 1e-6)
                    log_prob += math.log(prob)  # adding the log to avoid tiny ugly numbers
                class_scores[c] = log_prob
            predicted = max(class_scores, key=class_scores.get)
            predictions.append(predicted)
        return predictions
