import os
import csv
import openpyxl

BASE_DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data')

def load_glass_data():
    filepath = os.path.join(BASE_DATA_PATH, 'glass', 'glass.data')
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        data = [row for row in reader if row]  # discard useless empty lines

    # Convert all strings to float except the ID (0th index)
    data = [[float(x) for x in row] for row in data]

    for row in data:
        del row[0]  # no need for the ID column, Hend. Thank you.

    return data


def load_pump_data():
    filepath = os.path.join(BASE_DATA_PATH, 'pump', 'sensor.csv')
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        data = [row for row in reader if row]  # discard useless empty lines
    
    processed_data = []

    for i, row in enumerate(data):
        try:
            # 2:-1 coz the row is [id, timestamp, sensor1, ..., sensor52, status]
            # So we skip the id and timestamp, and the last column is the label otherwise -> convert to float
            sensor_values = [float(x) if x != '' else 0.0 for x in row[2:-1]]
            label = row[-1].strip()
            processed_data.append(sensor_values + [label])
        except ValueError:
            continue  # skip rows with invalid float conversion
        
    return header, processed_data


def load_energy_data():
    def convert_energy_xlsx_to_csv(xlsx_path, csv_path):
        workbook = openpyxl.load_workbook(xlsx_path)
        sheet = workbook.active

        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            for row in sheet.iter_rows(values_only=True):
                writer.writerow(row)

    
    xlsx_file = os.path.join(BASE_DATA_PATH, 'energy', 'ENB2012_data.xlsx')
    csv_file = os.path.join(BASE_DATA_PATH, 'energy', 'ENB2012_data.csv')

    # ONLY First time: Convert xlsx to csv if needed
    if not os.path.exists(csv_file):
        print("Converting Energy Efficiency Excel file to CSV...")
        convert_energy_xlsx_to_csv(xlsx_file, csv_file)
        print("Conversion done.")

    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skipping the header row

        data = []
        for row in reader:
            # We figured theire is an extra empty line in this dataset we need to ignore
            # Keeping only the first 10 columns [8 features + 2 targets]
            clean_row = [x for x in row if x.strip() != ''][:10]

            
            # Cleaning and skipping any rows with zero data
            if len(clean_row) != 10:
                continue
            try:
                row = [float(x) for x in clean_row]
                features = row[:8]           # 8 features
                target_heating = row[8]      # Y1
                target_cooling = row[9]      # Y2
                data.append(features + [target_heating, target_cooling])
            except ValueError:
                continue  # skip rows with invalid float conversion

    return data


def load_breast_cancer_data():
    filepath = os.path.join(BASE_DATA_PATH, 'breast_cancer', 'wdbc.data')
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        data = []
        for row in reader:
            try:
                #  Hend: So we want to use 0/1 classes so we can easily use this in the logistic regression model and the decision tree model 
                label = 1 if row[1] == 'M' else 0  # Malignant = 1, Benign = 0
                features = [float(x) for x in row[2:]]  # Skipping the id and label (0, 1)
                data.append(features + [label])
            except (ValueError, IndexError):
                continue  # skip bad rows
    return data


def load_mushroom_data():
    filepath = os.path.join(BASE_DATA_PATH, 'mushroom', 'mushroom.data')
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        data = []
        for row in reader:
            if len(row) > 1:
                features = row[1:]  # just skip the label
                label = row[0]
                data.append(features + [label])
    return data


def load_robot_data():
    folder = os.path.join(BASE_DATA_PATH, 'robot')
    all_data = []

    for i in range(1, 6):  # lp1 to lp5 
        filename = os.path.join(folder, f'lp{i}.data')
        with open(filename, 'r') as f:
            reader = csv.reader(f, delimiter=' ')
            for row in reader:
                row = [x for x in row if x.strip() != '']
                try:
                    features = [float(x) for x in row[:-1]]
                    label = row[-1]

                    all_data.append(features + [label])
                except ValueError:
                    continue  # skip rows with invalid float conversion

    return all_data

def load_robot_data():
    '''
    From the robot.data file we get: 
        5. Number of instances in each dataset
            -- LP1: 88
            -- LP2: 47
            -- LP3: 47
            -- LP4: 117
            -- LP5: 164

        6. Number of features: 90 (in any of the five datasets)

        7. Feature information
            -- All features are numeric (continuous, although integers only).
            -- Each feature represents a force or a torque measured after
                failure detection; each failure instance is characterized in terms
                of 15 force/torque samples collected at regular time intervals
                starting immediately after failure detection; 
                The total observation window for each failure instance was of 315 ms.
            -- Each example is described as follows:

                    class
                    Fx1	Fy1	Fz1	Tx1	Ty1	Tz1
                    Fx2	Fy2	Fz2	Tx2	Ty2	Tz2
                    ......
                    Fx15	Fy15	Fz15	Tx15	Ty15	Tz15

                where Fx1 ... Fx15 is the evolution of force Fx in the observation
                window, the same for Fy, Fz and the torques; there is a total 
                of 90 features.

    '''
    folder = os.path.join(BASE_DATA_PATH, 'robot')
    all_data = []

    for i in range(1, 6):  # lp1 to lp5 files
        filepath = os.path.join(folder, f'lp{i}.data')
        with open(filepath, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]

            idx = 0
            while idx < len(lines):
                # label line
                label = lines[idx].lower()
                idx += 1

                # Next 15 lines are the feature block
                feature_block = lines[idx:idx + 15]
                idx += 15

                if len(feature_block) < 15:
                    print("Incomplete example in Robot dataset.")
                    continue

                try:
                    # Flatten all 15 rows into a single list of 90 floats
                    features = []
                    for line in feature_block:
                        parts = [float(x) for x in line.split()]
                        features.extend(parts)

                    if len(features) == 90:
                        all_data.append(features + [label])
                except ValueError:
                    continue  # skip rows with invalid float conversion

    return all_data
