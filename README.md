# Machine Learning from Scratch

This project implements classic machine learning models entirely from scratch — without using libraries like `scikit-learn`. The models are applied to a variety of real-world datasets for both classification and regression tasks.

---

## Code Structure

```
ML_code/
│
├── data/                    # Contains raw and downloaded datasets
│   ├── glass/
│   ├── pump/
│   ├── energy/
│   └── ...
│
├── helpers/                # Utility modules
│   ├── data_loader.py      
│   ├── preprocessing.py    
│   ├── classification_metrics.py  # Accuracy, confusion matrix, etc.
│   ├── regression_metrics.py      # MSE, MAE, R^2
│
├── models/                 
│   ├── logistic_regression.py
│   ├── linear_regression.py
│   ├── naive_bayes_classifier.py
│   ├── gaussian_naive_bayes.py
│   ├── knn.py
│   └── decision_tree.py
│
├── experiments/            # Training/evaluation scripts per model -> What you should run
│   ├── run_logistic.py
│   ├── run_linear.py
│   ├── run_naive_bayes.py
│   ├── run_knn.py
│   └── run_decision_tree.py
│
├── history/                # Logs from training
|
|
├── plots/         # Have some of the plots from running the experiments
│
├── main.py             # just a file to check loading the databases before you run the experiments
|
├── hend_notes.txt     # Here is where I write my thoughts process through the project
|
├── requirements.txt         # Dependencies (minimal)
|
└── README.md
```

---

## Fun fact:
I wrote the hend_notes file in the begining of the project but as it became more complicated I had to let it go. I left it there for future references to how did I think.

---

## How to Run the Project

### 1. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install dependencies:
```bash
pip install -r requirements.txt
```

### 3. Run main to make sure the loading is working:
```bash
python main.py
```

### 4. Run experiments:
Each experiment script runs a specific model on all datasets.
```bash
python -m experiments.run_logistic
python -m experiments.run_linear
python -m experiments.run_naive_bayes
python -m experiments.run_knn
python -m experiments.run_decision_tree
```

---

## Datasets Used

1. **Glass Identification**
   - Multiclass classification (7 glass types)
   - All features are numeric

2. **Breast Cancer Diagnosis**
   - Binary classification (Malignant/Benign)
   - 30 numerical features from tumor measurements

3. **Mushroom Dataset**
   - Binary classification (Edible/Poisonous)
   - All features are categorical → one-hot encoded

4. **Pump Sensor Data**
   - Multiclass classification (NORMAL, BROKEN, RECOVERING)
   - Large dataset with 51 numeric features

5. **Robot Execution Failures**
   - Multiclass classification with 15 classes
   - High-dimensional (90 features), numeric

6. **Energy Efficiency**
   - Regression task (Heating Load prediction)
   - All features numeric; target is a continuous value (Y1)

---

## Evaluation & Metrics
- **Classification models:** Accuracy, Confusion Matrix, Per-class Accuracy Bar Charts
- **Regression model:** Mean Squared Error (MSE), Mean Absolute Error (MAE), R-squared (R²)
- Plots are in the `plots/` directory and logs are stored in the `history/` directory

---

## Notes
- The Robot dataset had to be manually downloaded due to broken archive links
- Each dataset is parsed uniquely depending on its format (e.g., `.data`, `.csv`, HTML tables)
- One-hot encoding and label encoding are handled in preprocessing

