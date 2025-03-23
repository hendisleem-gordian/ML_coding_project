import random
from collections import defaultdict

def train_test_split(data, test_ratio=0.2, seed=None):
    '''split the data into training and testing sets || Default: 80% training & 20% testing'''
    '''Fun fact: the seed parameter is used to generate the same sequence of "random" numbers if needed'''
    if seed is not None:
        random.seed(seed)

    shuffled = data.copy()
    random.shuffle(shuffled)

    split_idx = int(len(shuffled) * (1 - test_ratio))
    train = shuffled[:split_idx]
    test = shuffled[split_idx:]

    return train, test


def standardize(X):
    '''standardize the data'''
    '''Standardization is a scaling technique where the values are centered around the mean with a unit standard deviation.'''
    '''Hend: No single feature dominates due to large numeric ranges. HAHA'''
    num_features = len(X[0])
    means = [0.0] * num_features
    stds = [0.0] * num_features

    # calculating the means
    for i in range(num_features):
        feature_values = [row[i] for row in X]
        means[i] = sum(feature_values) / len(feature_values)

    # calculating the standard deviations
    for i in range(num_features):
        feature_values = [row[i] for row in X]
        stds[i] = (sum((x - means[i]) ** 2 for x in feature_values) / len(feature_values)) ** 0.5
        if stds[i] == 0:
            stds[i] = 1  # no /zero

    # standardization
    X_scaled = []
    for row in X:
        scaled_row = [(x - means[i]) / stds[i] for i, x in enumerate(row)]
        X_scaled.append(scaled_row)

    return X_scaled


def encode_labels(y):
    '''encode the labels given to it'''
    label_map = {}
    encoded = []

    current_label = 0
    for label in y:
        if label not in label_map:
            label_map[label] = current_label
            current_label += 1
        encoded.append(label_map[label])

    return encoded, label_map


def one_hot_encode_features(data):
    '''one hot encode the features for the mashroom dataset (talk about favoritism XD)'''
    num_features = len(data[0])
    unique_values = [set() for _ in range(num_features)]

    # First round to collect all unique values of each feature
    for row in data:
        for i, value in enumerate(row):
            unique_values[i].add(value)

    # Create sorted list of values per feature
    value_lists = [sorted(list(vals)) for vals in unique_values]
    index_maps = [{v: idx for idx, v in enumerate(vlist)} for vlist in value_lists]

    encoded_data = []

    for row in data:
        encoded_row = []
        for i, value in enumerate(row):
            one_hot = [0] * len(value_lists[i])
            if value in index_maps[i]:
                one_hot[index_maps[i][value]] = 1
            encoded_row.extend(one_hot)
        encoded_data.append(encoded_row)

    return encoded_data, index_maps
