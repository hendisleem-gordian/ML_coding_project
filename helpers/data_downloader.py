import os
import urllib.request
import subprocess

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data')


def download_pump_dataset():
    dest_folder = os.path.join(DATA_PATH, 'pump')
    os.makedirs(dest_folder, exist_ok=True)
    dest_file = os.path.join(dest_folder, 'sensor.csv')

    if not os.path.exists(dest_file):
        print("Downloading Pump Sensor dataset from Kaggle...")
        subprocess.run([
            'kaggle', 'datasets', 'download',
            '-d', 'nphantawee/pump-sensor-data',
            '--unzip',
            '-p', dest_folder
        ])
        print("Pump dataset downloaded.")
    else:
        print("Pump dataset already exists.")



def download_dataset(name, url, filename, subfolder):
    dest_folder = os.path.join(DATA_PATH, subfolder)
    os.makedirs(dest_folder, exist_ok=True)
    dest_file = os.path.join(dest_folder, filename)

    if not os.path.exists(dest_file):
        print(f"Downloading {name} dataset...")
        urllib.request.urlretrieve(url, dest_file)
        print(f"{name} dataset downloaded.")
    else:
        print(f"{name} dataset already exists.")


def download_glass_dataset():    
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data'
    download_dataset("Glass", url, 'glass.data', 'glass')


def download_energy_dataset():
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx'
    download_dataset("Energy Efficiency", url, 'ENB2012_data.xlsx', 'energy')


def download_breast_cancer_dataset():
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data'
    download_dataset("Breast Cancer", url, 'wdbc.data', 'breast_cancer')


def download_mushroom_dataset():
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data'
    download_dataset("Mushroom", url, 'mushroom.data', 'mushroom')


# def download_robot_dataset():
#     url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/robotexecution/robotExecutionFailure.zip'
#     download_dataset("Robot", url, 'robot.zip', 'robot')


def download_all():
    download_pump_dataset()
    download_glass_dataset()
    download_energy_dataset()
    download_breast_cancer_dataset()
    download_mushroom_dataset()
    # download_robot_dataset() -> Link is wrong? urllib.error.HTTPError: HTTP Error 404: Not Found

if __name__ == "__main__":
    download_all()
