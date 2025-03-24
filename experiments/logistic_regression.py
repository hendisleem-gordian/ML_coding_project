from helpers.classification_metrics import accuracy_score, print_confusion_matrix
from helpers.data_loader import load_glass_data, load_pump_data, load_breast_cancer_data, load_mushroom_data, load_robot_data
from helpers.preprocessing import encode_labels, one_hot_encode_features, train_test_split, standardize
import matplotlib.pyplot as plt

from models.logistic_regression import MulticlassLogisticRegression



def pre_processing(data_name, x, y):
    train, test = train_test_split(list(zip(x, y)), seed=42)
    x_train, y_train = zip(*train)
    x_test, y_test = zip(*test)
    x_train = standardize(x_train)
    x_test = standardize(x_test)
    print(f"\n{data_name}:", len(x_train), "train /", len(x_test), "test")

    return x_train, y_train, x_test, y_test



def train_dataset(data_name, x, y, lr, epochs):
    x_train, y_train, x_test, y_test = pre_processing("Glass", x, y)
    model = MulticlassLogisticRegression(lr, epochs)
    model.fit(x_train, y_train)
    preds = model.predict(x_test)

    # Evaluation 
    acc = accuracy_score(y_test, preds)
    print(f">>>>>>>>  {data_name} Accuracy: {acc:.4f}")
    print_confusion_matrix(y_test, preds)

    # Plot accuracy over epochs
    plt.plot(model.training_accuracy, label="Accuracy")
    plt.plot(model.training_loss, label="Loss")
    plt.xlabel("Epoch")
    plt.title(f"Training Progress for {data_name} dataset.")
    plt.legend()
    plt.grid(True)
    plt.show()


def run_logistic_regression():
    
    # # GLASS (multiclass)
    glass = load_glass_data()
    x_glass = [row[:-1] for row in glass]
    y_glass, glass_label_map = encode_labels([row[-1] for row in glass])

    train_dataset("Glass", x_glass, y_glass, lr=0.005, epochs=3000)

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    
    # # PUMP (multiclass)
    header, pump = load_pump_data()
    x_pump = [row[:-1] for row in pump]
    y_pump, pump_label_map = encode_labels([row[-1] for row in pump])

    train_dataset("Pump", x_pump, y_pump,lr=0.01, epochs=1000)

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    
    # # BREAST CANCER (binary)
    bc = load_breast_cancer_data()
    x_bc = [row[:-1] for row in bc]
    y_bc = [int(row[-1]) for row in bc]

    train_dataset("Breast Cancer", x_bc, y_bc, lr=0.01, epochs=1000)

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    
    # MUSHROOM (binary)
    mushroom = load_mushroom_data()
    x_mushroom, index_maps = one_hot_encode_features([row[:-1] for row in mushroom])
    y_mushroom, mushroom_label_map = encode_labels([row[-1] for row in mushroom])

    train_dataset("Mushroom", x_mushroom, y_mushroom, lr=0.01, epochs=1000)

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    
    # ROBOT (multiclass)
    robot = load_robot_data()
    x_robot = [row[:-1] for row in robot]
    y_robot, robot_label_map = encode_labels([row[-1] for row in robot])

    train_dataset("Robot", x_robot, y_robot, lr=0.01, epochs=1000)

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------

if __name__ == "__main__":
    run_logistic_regression()
