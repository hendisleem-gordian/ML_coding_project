from helpers.classification_metrics import accuracy_score, print_confusion_matrix
from helpers.data_loader import load_glass_data, load_pump_data, load_breast_cancer_data, load_mushroom_data, load_robot_data
from helpers.preprocessing import encode_labels, one_hot_encode_features, train_test_split, standardize
import matplotlib.pyplot as plt

from models.naive_bayes_classifier  import NaiveBayesClassifier

from collections import defaultdict

def pre_processing(data_name, x, y):
    train, test = train_test_split(list(zip(x, y)), seed=42)
    x_train, y_train = zip(*train)
    x_test, y_test = zip(*test)
    print(f"\n{data_name}:", len(x_train), "train /", len(x_test), "test")

    return x_train, y_train, x_test, y_test


def plot_per_class_accuracy(data_name, y_true, y_pred, label_names=None):
    correct = defaultdict(int)
    total = defaultdict(int)

    for yt, yp in zip(y_true, y_pred):
        total[yt] += 1
        if yt == yp:
            correct[yt] += 1

    classes = sorted(total.keys())
    accuracies = [correct[c] / total[c] for c in classes]

    # If label_names provided, use them for x-ticks
    if label_names:
        xticks = [label_names[c] for c in classes]
    else:
        xticks = classes

    plt.bar(xticks, accuracies)
    plt.xlabel("Class")
    plt.ylabel("Accuracy")
    plt.title(f"Per-Class Accuracy for {data_name} dataset.")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.show()



def train_dataset_with_naive_bayes(data_name, x, y, original_label_map=None):
    x_train, y_train, x_test, y_test = pre_processing(data_name, x, y)
    model = NaiveBayesClassifier()
    model.fit(x_train, y_train)
    preds = model.predict(x_test)

    # Evaluation 
    acc = accuracy_score(y_test, preds)
    print(f">>>>>>>>  {data_name} Accuracy: {acc:.4f}")
    print_confusion_matrix(y_test, preds)

    # Plot how accurate the model is for each class individually
    if original_label_map:
        reverse_label_map = {v: k for k, v in original_label_map.items()}
        label_names = [reverse_label_map[c] for c in sorted(reverse_label_map.keys())]
        plot_per_class_accuracy(data_name, y_test, preds, label_names=label_names)
    else:
        plot_per_class_accuracy(data_name, y_test, preds)


def run_naive_bayes():
    
    # GLASS (multiclass)
    glass = load_glass_data()
    x_glass = [row[:-1] for row in glass]
    y_glass, glass_label_map = encode_labels([row[-1] for row in glass])

    train_dataset_with_naive_bayes("Glass", x_glass, y_glass, glass_label_map)

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    
    # PUMP (multiclass)
    header, pump = load_pump_data()
    x_pump = [row[:-1] for row in pump]
    y_pump, pump_label_map = encode_labels([row[-1] for row in pump])

    train_dataset_with_naive_bayes("Pump", x_pump, y_pump, pump_label_map)

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    
    # # BREAST CANCER (binary)
    bc = load_breast_cancer_data()
    x_bc = [row[:-1] for row in bc]
    y_bc = [int(row[-1]) for row in bc]
    label_map = {'B': 0, 'M': 1}

    train_dataset_with_naive_bayes("Breast Cancer", x_bc, y_bc, label_map)

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    
    # MUSHROOM (binary)
    mushroom = load_mushroom_data()
    x_mushroom, index_maps = one_hot_encode_features([row[:-1] for row in mushroom])
    y_mushroom, mushroom_label_map = encode_labels([row[-1] for row in mushroom])

    train_dataset_with_naive_bayes("Mushroom", x_mushroom, y_mushroom, mushroom_label_map)

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    
    # ROBOT (multiclass)
    robot = load_robot_data()
    x_robot = [row[:-1] for row in robot]
    y_robot, robot_label_map = encode_labels([row[-1] for row in robot])

    train_dataset_with_naive_bayes("Robot", x_robot, y_robot, robot_label_map)

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------

if __name__ == "__main__":
    run_naive_bayes()
