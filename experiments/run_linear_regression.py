from helpers import regression_metrics
from helpers.data_loader import load_energy_data
from helpers.preprocessing import train_test_split, standardize
import matplotlib.pyplot as plt

from models.linear_regression import LinearRegression



def pre_processing(data_name, x, y):
    train, test = train_test_split(list(zip(x, y)), seed=42)
    x_train, y_train = zip(*train)
    x_test, y_test = zip(*test)
    x_train = standardize(x_train)
    x_test = standardize(x_test)
    print(f"\n{data_name}:", len(x_train), "train /", len(x_test), "test")

    return x_train, y_train, x_test, y_test



def train_dataset(data_name, x, y, lr, epochs):
    x_train, y_train, x_test, y_test = pre_processing(data_name, x, y)
    model = LinearRegression(lr, epochs)
    model.fit(x_train, y_train)
    preds = model.predict(x_test)

    # Plot the loss
    plt.plot(model.training_loss)
    plt.title(f"Linear Regression Training Loss for {data_name} dataset.")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True)
    plt.show()

    # Evaluation 
    mse = regression_metrics.mean_squared_error(y_test, preds)  # average squared difference between predicted and true values. 
    mae = regression_metrics.mean_absolute_error(y_test, preds)  # average distance between predictions and true values (in original units). 
    r2 = regression_metrics.r2_score(y_test, preds)  # percentage of variance in the target that your model explains.

    print(f"MSE: {mse:.4f}")  # -> “How badly do I mess up — especially when I'm way off?”
    print(f"MAE: {mae:.4f}")  # -> “On average, how far off am I?”
    print(f"R²: {r2:.4f}")    # -> “How well do I explain the true data?”

    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, preds, alpha=0.6)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    plt.xlabel("Actual Y")
    plt.ylabel("Predicted Y")
    plt.title("Actual vs Predicted (Y)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def run_linear_regression():
    
    # ENERGY (Regression)
    energy = load_energy_data()
    x_energy = [row[:-2] for row in energy] 
    y1_energy = [row[-2] for row in energy]  # Y1 -> Heating Load
    # y2_energy = [row[-1] for row in energy]  # Y2 -> Cooling load

    train_dataset("nEnergy", x_energy, y1_energy, lr=0.005, epochs=3000)

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    
if __name__ == "__main__":
    run_linear_regression()
