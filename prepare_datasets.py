from helpers.data_loader import load_glass_data, load_pump_data, load_energy_data, load_breast_cancer_data, load_mushroom_data, load_robot_data
from helpers.preprocessing import encode_labels, one_hot_encode_features, train_test_split, standardize
import pprint
import matplotlib.pyplot as plt

from models.logistic_regression import LogisticRegression, MulticlassLogisticRegression

pp = pprint.PrettyPrinter(indent=2)


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# GLASS (Multiclass classification)
# glass = load_glass_data()
# # print(f"Loaded Glass dataset with {len(glass)} samples and {len(glass[0]) - 1} features + 1 label.")
# # print("Sample row:", glass[0])
# X_glass = [row[:-1] for row in glass]
# # y_glass = [int(row[-1]) for row in glass]

# y_glass, pump_label_map = encode_labels([row[-1] for row in glass])

# glass_train, glass_test = train_test_split(list(zip(X_glass, y_glass)), seed=42)
# Xg_train, yg_train = zip(*glass_train)
# Xg_test, yg_test = zip(*glass_test)
# Xg_train = standardize(Xg_train)
# Xg_test = standardize(Xg_test)
# print("\nGlass:", len(Xg_train), "train /", len(Xg_test), "test")
# pp.pprint(Xg_train[0])
# print("Label:", yg_train[0])


# # model = MulticlassLogisticRegression(lr=0.01, epochs=1000)

# model = MulticlassLogisticRegression(lr=0.005, epochs=3000)
# # model = MulticlassLogisticRegression(lr=0.01, epochs=5000)
# # model = MulticlassLogisticRegression(lr=0.05, epochs=1500)



# model.fit(Xg_train, yg_train)
# preds = model.predict(Xg_test)

# accuracy = sum(1 for p, y in zip(preds, yg_test) if p == y) / len(yg_test)
# print(f"🎯 Glass Accuracy: {accuracy:.4f}")

# # Plot accuracy over epochs
# plt.plot(model.training_accuracy, label="Accuracy")
# plt.plot(model.training_loss, label="Loss")
# plt.xlabel("Epoch")
# plt.title("Training Progress")
# plt.legend()
# plt.grid(True)
# plt.show()

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


# # PUMP SENSOR (Multiclass classification)
# header, pump = load_pump_data()
# # print(f"\nLoaded Pump dataset with {len(pump)} samples and {len(pump[0]) - 2} features + 1 label.")
# # print("Header:", header)
# # print("Sample row:", pump[0])
# X_pump = [row[:-1] for row in pump]
# y_pump, pump_label_map = encode_labels([row[-1] for row in pump])
# pump_train, pump_test = train_test_split(list(zip(X_pump, y_pump)), seed=42)
# Xp_train, yp_train = zip(*pump_train)
# Xp_test, yp_test = zip(*pump_test)
# Xp_train = standardize(Xp_train)
# Xp_test = standardize(Xp_test)

# print("\nPump Sensor:", len(Xp_train), "train /", len(Xp_test), "test")



# model = NewMulticlassLogisticRegression(lr=0.01, epochs=1000)
# model.fit(Xp_train, yp_train)
# preds = model.predict(Xp_test)

# acc = sum(1 for p, y in zip(preds, yp_test) if p == y) / len(yp_test)
# print(f"\n🎯 Pump Sensor Accuracy: {acc:.4f}")

# # Plot accuracy over epochs
# plt.plot(model.training_accuracy, label="Accuracy")
# plt.plot(model.training_loss, label="Loss")
# plt.xlabel("Epoch")
# plt.title("Training Progress")
# plt.legend()
# plt.grid(True)
# plt.show()


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


"""
# ENERGY (Regression)
energy = load_energy_data()
# print(f"\nLoaded Energy dataset with {len(energy)} samples and {len(energy[0]) - 2} features + 2 targets.")
# print("Sample row:", energy[0])
X_energy = [row[:-2] for row in energy]  # First 8 columns -> features
y_energy = [row[-2] for row in energy]   # Y1 -> Heating Load
# y_energy = [row[-1] for row in energy]  # Y2 -> Cooling load
energy_train, energy_test = train_test_split(list(zip(X_energy, y_energy)), seed=42)
Xe_train, ye_train = zip(*energy_train)
Xe_test, ye_test = zip(*energy_test)
Xe_train = standardize(Xe_train)
Xe_test = standardize(Xe_test)

print("\nEnergy:", len(Xe_train), "train /", len(Xe_test), "test")
"""


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------



# BREAST CANCER (Binary classification)
bc_data = load_breast_cancer_data()
# print(f"Breast Cancer: {len(bc_data)} samples, {len(bc_data[0]) - 1} features")
# print("Sample row:", bc_data[0])
X_bc = [row[:-1] for row in bc_data]
# y_bc = [row[-1] for row in bc_data]

y_bc, bc_label_map = encode_labels([row[-1] for row in bc_data])
bc_train, bc_test = train_test_split(list(zip(X_bc, y_bc)), seed=42)
Xbc_train, ybc_train = zip(*bc_train)
Xbc_test, ybc_test = zip(*bc_test)
Xbc_train = standardize(Xbc_train)
Xbc_test = standardize(Xbc_test)
print("\nBreast Cancer:", len(Xbc_train), "train /", len(Xbc_test), "test")



# Use already-preprocessed breast cancer data
# model = LogisticRegression(lr=0.01, epochs=1000)
model = MulticlassLogisticRegression(lr=0.01, epochs=1000)
model.fit(Xbc_train, ybc_train)
preds = model.predict(Xbc_test)

# Accuracy
acc = sum(1 for p, y in zip(preds, ybc_test) if p == y) / len(ybc_test)
print(f"\n🎯 Breast Cancer Accuracy: {acc:.4f}")






# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------



# # MUSHROOM (Binary classification, categorical)
# mush = load_mushroom_data()
# # print(f"Mushroom: {len(mush)} samples, {len(mush[0]) - 1} features")
# # print("Sample row:", mush[0])
# # Separate features and labels
# X_mush_raw = [row[:-1] for row in mush]
# y_mush_raw = [row[-1] for row in mush]

# # One-hot encode features
# X_mush, _ = one_hot_encode_features(X_mush_raw)
# # Encode labels ('p' or 'e')
# y_mush, mushroom_label_map = encode_labels(y_mush_raw)

# # Train/test split
# mush_train, mush_test = train_test_split(list(zip(X_mush, y_mush)), seed=42)
# Xm_train, ym_train = zip(*mush_train)
# Xm_test, ym_test = zip(*mush_test)

# print("\nMushroom:", len(Xm_train), "train /", len(Xm_test), "test")

# model = MulticlassLogisticRegression(lr=0.01, epochs=1000)
# model.fit(Xm_train, ym_train)
# preds = model.predict(Xm_test)

# acc = sum(1 for p, y in zip(preds, ym_test) if p == y) / len(ym_test)
# print(f"🎯 Mushroom Accuracy: {acc:.4f}")






# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------




# # ROBOT EXECUTION FAILURES (Multiclass classification)
# robot = load_robot_data()
# # print(f"Robot: {len(robot)} samples, {len(robot[0]) - 1} features")
# # print("Sample row:", robot[0])
# X_robot = [row[:-1] for row in robot]
# y_robot, robot_label_map = encode_labels([row[-1] for row in robot])
# robot_train, robot_test = train_test_split(list(zip(X_robot, y_robot)), seed=42)
# Xr_train, yr_train = zip(*robot_train)
# Xr_test, yr_test = zip(*robot_test)
# Xr_train = standardize(Xr_train)
# Xr_test = standardize(Xr_test)

# print("\nRobot:", len(Xr_train), "train /", len(Xr_test), "test")

# model = MulticlassLogisticRegression(lr=0.01, epochs=1000)
# model.fit(Xr_train, yr_train)
# preds = model.predict(Xr_test)

# acc = sum(1 for p, y in zip(preds, yr_test) if p == y) / len(yr_test)
# print(f"🎯 Robot Accuracy: {acc:.4f}")

""""""