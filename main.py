from helpers.data_loader import load_glass_data, load_pump_data, load_energy_data, load_breast_cancer_data, load_mushroom_data, load_robot_data

glass = load_glass_data()
print(f"Loaded Glass dataset with {len(glass)} samples and {len(glass[0]) - 1} features + 1 label.")
print("Sample row:", glass[0])

header, pump = load_pump_data()
print(f"\nLoaded Pump dataset with {len(pump)} samples and {len(pump[0]) - 2} features + 1 label.")
print("Header:", header)
print("Sample row:", pump[0])

energy = load_energy_data()
print(f"\nLoaded Energy dataset with {len(energy)} samples and {len(energy[0]) - 2} features + 2 targets.")
print("Sample row:", energy[0])

bc = load_breast_cancer_data()
print(f"Breast Cancer: {len(bc)} samples, {len(bc[0]) - 1} features")
print("Sample row:", bc[0])

mush = load_mushroom_data()
print(f"Mushroom: {len(mush)} samples, {len(mush[0]) - 1} features")
print("Sample row:", mush[0])

robot = load_robot_data()
print(f"Robot: {len(robot)} samples, {len(robot[0]) - 1} features")
print("Sample row:", robot[0])
