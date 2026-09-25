# ❤️ Heart Disease Prediction using Machine Learning

An end-to-end machine learning project that predicts the likelihood of heart disease from patient health and clinical parameters.

The project covers the complete ML workflow — **data exploration, cleaning, preprocessing, model comparison, feature scaling, model serialization, and deployment through a Streamlit web application**.

> **Disclaimer:** This project is created for educational and portfolio purposes. It is not a medical diagnostic system and should not be used as a substitute for professional medical advice.

---

## 🚀 Project Overview

The application accepts patient information such as age, sex, chest-pain type, resting blood pressure, cholesterol, maximum heart rate, exercise-induced angina, and other clinical parameters.

The trained machine learning model then classifies the input into:

- `0` → Low Risk of Heart Disease
- `1` → High Risk of Heart Disease

### End-to-End Workflow

```text
Heart Disease Dataset
        │
        ▼
Exploratory Data Analysis
        │
        ▼
Data Cleaning
        │
        ├── Replace invalid zero values
        └── Handle categorical variables
        │
        ▼
One-Hot Encoding
        │
        ▼
Train / Test Split
        │
        ▼
StandardScaler
        │
        ▼
Compare Multiple ML Models
        │
        ▼
Logistic Regression
        │
        ├── Save trained model
        ├── Save scaler
        └── Save expected feature columns
        │
        ▼
Streamlit Application
        │
        ▼
Heart Disease Risk Prediction
```

---

## ✨ Key Features

- Interactive Streamlit prediction interface
- Logistic Regression classification model
- Exploratory Data Analysis using Pandas, Matplotlib and Seaborn
- Categorical feature encoding using One-Hot Encoding
- Numerical feature standardization using `StandardScaler`
- Comparison of multiple classification algorithms
- Persisted ML artifacts using Joblib
- Consistent feature ordering during inference
- Simple local and cloud deployment structure

---

## 📊 Dataset

The project uses a heart disease dataset containing **918 records and 12 columns**.

### Input Features

| Feature | Description |
|---|---|
| `Age` | Age of the patient |
| `Sex` | Sex of the patient |
| `ChestPainType` | Type of chest pain |
| `RestingBP` | Resting blood pressure |
| `Cholesterol` | Cholesterol level |
| `FastingBS` | Fasting blood sugar indicator |
| `RestingECG` | Resting electrocardiogram result |
| `MaxHR` | Maximum heart rate achieved |
| `ExerciseAngina` | Exercise-induced angina |
| `Oldpeak` | ST depression |
| `ST_Slope` | Slope of the peak exercise ST segment |
| `HeartDisease` | Target variable |

### Target Distribution

- Heart Disease = `1`: 508 records
- Heart Disease = `0`: 410 records

---

## 🧹 Data Preprocessing

The training pipeline performs the following preprocessing steps:

### 1. Data Inspection

The dataset is explored using:

- Shape and column inspection
- Descriptive statistics
- Duplicate checking
- Distribution plots
- Categorical feature analysis
- Correlation analysis

### 2. Handling Invalid Zero Values

Zero values in `Cholesterol` and `RestingBP` are treated as invalid/missing-like values and replaced with the mean calculated from their non-zero observations.

### 3. Categorical Encoding

Categorical features are converted into numerical features using:

```python
pd.get_dummies(df, drop_first=True)
```

### 4. Train-Test Split

The dataset is divided into:

- 67% training data
- 33% testing data

with:

```python
random_state=42
```

### 5. Feature Scaling

`StandardScaler` is fitted on the training data and then applied to both training and testing data.

This same fitted scaler is saved and reused during prediction so that inference uses the same transformation as training.

---

## 🤖 Machine Learning Models

Five classification algorithms were evaluated:

1. Logistic Regression
2. K-Nearest Neighbors
3. Gaussian Naive Bayes
4. Decision Tree
5. Support Vector Machine

### Model Comparison

| Model | Accuracy | F1 Score |
|---|---:|---:|
| Logistic Regression | 87.13% | 88.70% |
| K-Nearest Neighbors | 84.49% | 86.30% |
| Gaussian Naive Bayes | 85.81% | 87.46% |
| Decision Tree | 73.60% | 75.90% |
| Support Vector Machine | 86.14% | 88.00% |

Based on the evaluation performed in the training script, **Logistic Regression was selected for the deployed application**.

> These metrics come from a single train/test split and should not be interpreted as clinical performance. Cross-validation and external validation would be appropriate for a production-grade medical ML system.

---

## 📦 Saved Model Artifacts

The trained pipeline stores three artifacts in the `Backend` directory:

### `logistic_regression_model.pkl`

The trained Logistic Regression classifier.

### `scaler.pkl`

The fitted `StandardScaler` used to standardize input features.

### `columns.pkl`

The exact feature-column order expected by the trained model.

Keeping the expected columns is important because the Streamlit application must generate the same feature representation used during training.

---

## 🖥️ Streamlit Application

The frontend is implemented using Streamlit.

The application collects:

- Age
- Sex
- Chest Pain Type
- Resting Blood Pressure
- Cholesterol
- Fasting Blood Sugar
- Resting ECG
- Maximum Heart Rate
- Exercise-Induced Angina
- Oldpeak
- ST Slope

After clicking **Predict**, the application:

1. Builds a DataFrame from the user's input.
2. Creates the required encoded categorical columns.
3. Adds any missing expected columns with `0`.
4. Reorders columns according to `columns.pkl`.
5. Applies the saved scaler.
6. Sends the transformed data to the trained Logistic Regression model.
7. Displays the prediction.

---

## 📁 Project Structure

```text
Heart_Disease_Project/
│
├── Backend/
│   ├── heart.csv
│   ├── heart_disease_prediction_training.py
│   ├── logistic_regression_model.pkl
│   ├── scaler.pkl
│   └── columns.pkl
│
├── Frontend/
│   └── app.py
│
└── README.md
```

---

## 🛠️ Technologies Used

### Programming Language

- Python

### Machine Learning

- Scikit-learn
- Logistic Regression
- KNN
- Gaussian Naive Bayes
- Decision Tree
- SVM
- StandardScaler

### Data Processing

- Pandas
- NumPy

### Data Visualization

- Matplotlib
- Seaborn

### Model Persistence

- Joblib

### Deployment / UI

- Streamlit

---

# ⚙️ Run the Project Locally

## 1. Clone the Repository

Clone the repository and navigate to the project:

```bash
git clone <repository-url>
cd Heart_Disease_Project
```

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib streamlit
```

## 4. Run the Streamlit Application

From the `Frontend` directory:

```bash
cd Frontend
streamlit run app.py
```

The application will open in your browser.

---

# 🧪 Train the Model Again

If you want to reproduce the training process, navigate to the `Backend` directory:

```bash
cd Backend
python heart_disease_prediction_training.py
```

The script will:

- Load the dataset
- Perform exploratory analysis
- Clean the data
- Encode categorical variables
- Split the dataset
- Scale features
- Train and evaluate multiple models
- Save the Logistic Regression model
- Save the scaler
- Save the expected feature columns

The generated artifacts are:

```text
logistic_regression_model.pkl
scaler.pkl
columns.pkl
```

---

# ☁️ Deployment

The application can be deployed on a platform that supports Streamlit applications.

Recommended deployment flow:

```text
GitHub Repository
       │
       ▼
Streamlit-compatible deployment platform
       │
       ▼
Install Python dependencies
       │
       ▼
Run Frontend/app.py
       │
       ▼
Public Web Application
```

### Important Deployment Note

The current `app.py` loads model files using relative paths:

```python
r'..\Backend\logistic_regression_model.pkl'
r'..\Backend\scaler.pkl'
r'..\Backend\columns.pkl'
```

For more reliable deployment across different environments, it is recommended to construct paths relative to the location of `app.py`, rather than depending on the directory from which Streamlit is launched.

For example:

```python
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
BACKEND_DIR = BASE_DIR.parent / "Backend"

model = joblib.load(BACKEND_DIR / "logistic_regression_model.pkl")
scaler = joblib.load(BACKEND_DIR / "scaler.pkl")
expected_columns = joblib.load(BACKEND_DIR / "columns.pkl")
```

This makes the application less dependent on the current working directory.

---

# 🔐 Production Considerations

For a production-quality ML application, the following improvements would be recommended:

- Use cross-validation instead of relying on a single train/test split.
- Evaluate precision, recall, ROC-AUC and confusion matrix in addition to accuracy and F1.
- Perform external validation on an independent dataset.
- Use a reproducible preprocessing pipeline such as `Pipeline` and `ColumnTransformer`.
- Version the dataset and trained model.
- Pin dependency versions in `requirements.txt`.
- Add automated tests for preprocessing and prediction.
- Add input validation and error handling.
- Monitor model performance after deployment.
- Add model and data version tracking.
- Consider probability calibration if risk probabilities are displayed.
- Avoid presenting predictions as medical diagnoses.

---

# 🔮 Future Improvements

- Add probability/risk score visualization.
- Add ROC-AUC and confusion matrix visualization.
- Introduce cross-validation and hyperparameter tuning.
- Build a reusable Scikit-learn preprocessing pipeline.
- Add automated unit tests.
- Add `requirements.txt`.
- Add Docker support.
- Add CI/CD using GitHub Actions.
- Improve Streamlit UI and accessibility.
- Add model explainability using SHAP or similar techniques.
- Add model/data versioning.
- Deploy the application publicly.

---

# 🎯 What This Project Demonstrates

This project demonstrates practical understanding of:

- End-to-end Machine Learning workflow
- Exploratory Data Analysis
- Data cleaning
- Feature engineering
- Categorical encoding
- Feature scaling
- Classification algorithms
- Model comparison
- Model evaluation
- Model serialization
- Consistent training/inference preprocessing
- Streamlit application development
- ML model deployment concepts

---

## 👨‍💻 Author

**Arbaaz Shaikh**

Software Developer | Java | Spring Boot | React | Machine Learning

GitHub: `arbaaz65dac`

---

## 📌 Disclaimer

This application is intended only for **educational, demonstration, and portfolio purposes**.

The predictions generated by this project should **not be used to diagnose, treat, or make medical decisions about heart disease**. Real-world clinical applications require appropriate medical validation, regulatory compliance, clinical oversight, and evaluation on representative independent data.
