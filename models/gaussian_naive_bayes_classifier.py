import math
from collections import defaultdict, Counter


class GaussianNaiveBayes:
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
        self.feature_stats = {}

        for c in class_counts:
            class_samples = [x_ for x_, lbl in zip(x, y) if lbl == c]
            stats = []
            for i in range(n_features):
                feature_values = [x[i] for x in class_samples]
                mean = sum(feature_values) / len(feature_values)
                var = sum((x - mean) ** 2 for x in feature_values) / len(feature_values)
                stats.append((mean, var))
            self.feature_stats[c] = stats

    def _gaussian_prob(self, x, mean, var):
        if var == 0:
            return 1e-9 if x != mean else 1.0
        exponent = math.exp(-((x - mean) ** 2) / (2 * var))
        return (1 / math.sqrt(2 * math.pi * var)) * exponent
    
    def predict(self, x):
        predictions = []
        for row in x:
            class_probs = {}
            for label, prior in self.class_priors.items():
                log_prob = math.log(prior)
                for i in range(len(row)):
                    mean, var = self.feature_stats[label][i]
                    prob = self._gaussian_prob(row[i], mean, var)
                    log_prob += math.log(prob + 1e-9) 
                class_probs[label] = log_prob
            predictions.append(max(class_probs, key=class_probs.get))
        return predictions
