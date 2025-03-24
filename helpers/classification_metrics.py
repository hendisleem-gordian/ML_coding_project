
from collections import Counter, defaultdict


def accuracy_score(y_true, y_pred):
    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    return correct / len(y_true)

def print_confusion_matrix(y_true, y_pred):
    classes = sorted(set(y_true) | set(y_pred))
    matrix = defaultdict(lambda: [0] * len(classes))
    class_to_idx = {label: idx for idx, label in enumerate(classes)}

    for yt, yp in zip(y_true, y_pred):
        matrix[yt][class_to_idx[yp]] += 1

    header = "     " + "  ".join(str(c) for c in classes)
    print("Confusion Matrix:")
    print(header)
    for c in classes:
        row = f"{str(c):>3}: " + "  ".join(f"{matrix[c][class_to_idx[j]]:>2}" for j in classes)
        print(row)

    return matrix