# Explainable AI-Based Loan Approval Prediction System

An **Explainable AI (XAI) based machine learning system** that predicts whether a loan application is likely to be approved or rejected and explains the factors influencing the prediction using **SHAP (SHapley Additive exPlanations)**.

## 🚀 Live Application

The application is built using **Streamlit** and can be deployed as a web application.

> Live demo: https://explainable-ai-based-loan-approval-prediction-system-wgemo7y2s.streamlit.app/

## 📌 Project Overview

Traditional loan approval systems can provide a prediction without clearly explaining why a particular decision was made.

<img width="1872" height="896" alt="Screenshot 2026-08-25 144126" src="https://github.com/user-attachments/assets/83685a05-0e55-4a81-bd15-8ca63e4d8319" />


This project addresses that problem by combining:

* Machine Learning for loan approval prediction
* XGBoost for high-performance classification
* Probability-based prediction
* SHAP for model explainability
* Streamlit for an interactive web interface

The system allows users to enter applicant information and receive:

<img width="1795" height="692" alt="Screenshot 2026-08-25 144420" src="https://github.com/user-attachments/assets/3bc28b7c-59c2-4ac3-9ee4-2b0f87fbcf05" />

1. Loan approval/rejection prediction
2. Approval probability
3. Rejection probability
4. Feature-level explanation of the prediction
   <img width="1732" height="854" alt="Screenshot 2026-08-25 144445" src="https://github.com/user-attachments/assets/4236a217-bedb-4b7b-9bc8-ecf438ccb521" />


## 🎯 Objectives

* Build a machine learning model for loan approval prediction.
* Compare multiple classification algorithms.
* Select the best-performing model.
* Provide probability-based predictions.
* Explain individual predictions using SHAP.
* Deploy the model through an interactive web application.

## 🧠 Machine Learning Models

The project evaluates multiple classification algorithms:

* Logistic Regression
* Random Forest
* XGBoost

### Model Performance

| Model               |   Accuracy |  Precision |     Recall |   F1 Score |
| ------------------- | ---------: | ---------: | ---------: | ---------: |
| **XGBoost**         | **98.59%** | **98.69%** | **99.06%** | **98.87%** |
| Random Forest       |     98.24% |     98.31% |     98.87% |     98.59% |
| Logistic Regression |     80.33% |     79.42% |     92.28% |     85.37% |

Based on the evaluation results, **XGBoost** was selected as the final model.

## 📊 Dataset Features

The model uses the following applicant attributes:

| Feature                    | Description                 |
| -------------------------- | --------------------------- |
| `no_of_dependents`         | Number of dependents        |
| `education`                | Applicant education status  |
| `self_employed`            | Self-employment status      |
| `income_annum`             | Annual income               |
| `loan_amount`              | Requested loan amount       |
| `loan_term`                | Loan repayment term         |
| `cibil_score`              | Applicant CIBIL score       |
| `residential_assets_value` | Value of residential assets |
| `commercial_assets_value`  | Value of commercial assets  |
| `luxury_assets_value`      | Value of luxury assets      |
| `bank_asset_value`         | Value of bank assets        |

The `loan_id` field is used as an identifier and is not used as a predictive feature.

## 🔍 Explainable AI

The project uses **SHAP** to explain individual XGBoost predictions.

SHAP values indicate how each feature contributes to the model's decision.

### Interpretation

* **Positive SHAP value** → pushes the prediction toward approval.
* **Negative SHAP value** → pushes the prediction toward rejection.
* Larger absolute SHAP values indicate stronger influence.

For example, a high CIBIL score can have a strong positive contribution toward loan approval, while certain loan characteristics can contribute negatively.

This makes the model more transparent and easier to interpret than a prediction-only system.

## 🌐 Streamlit Application

The Streamlit application provides an interactive interface where users can enter:

* Number of dependents
* Education
* Employment status
* Annual income
* Loan amount
* Loan term
* CIBIL score
* Residential assets
* Commercial assets
* Luxury assets
* Bank assets

The application then displays:

### Prediction

**Approved / Rejected**

### Probability

* Approval probability
* Rejection probability

### Explainability

A SHAP-based feature contribution graph showing the most influential factors behind the prediction.

## 🏗️ Project Structure

```text
Explainable-AI-Based-Loan-Approval-Prediction-System/
│
├── app.py
├── requirements.txt
│
├── notebooks/
│   ├── 01_data_analysis.ipynb
│   └── xgb_pipeline.pkl
│
└── README.md
```

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/anubhav7790/Explainable-AI-Based-Loan-Approval-Prediction-System.git
```

Move into the project directory:

```bash
cd Explainable-AI-Based-Loan-Approval-Prediction-System
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## ▶️ Run the Application

Start Streamlit:

```bash
streamlit run app.py
```

The application will open locally in your browser.

## 🛠️ Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **XGBoost**
* **SHAP**
* **Matplotlib**
* **Streamlit**
* **Joblib**
* **Jupyter Notebook**

## 📈 Evaluation

The final XGBoost model achieved approximately:

* **98.59% Accuracy**
* **98.69% Precision**
* **99.06% Recall**
* **98.87% F1 Score**

The model was evaluated using a held-out test set.

## 🔮 Future Improvements

Potential future improvements include:

* Hyperparameter optimization
* Cross-validation
* More advanced SHAP visualizations
* Model fairness and bias analysis
* User authentication
* Database integration
* Loan risk scoring
* Cloud deployment
* Monitoring model performance after deployment

## ⚠️ Disclaimer

This project is intended for **educational and research purposes**. It should not be used as the sole basis for real-world financial or lending decisions.

## 👨‍💻 Author

**Anubhav Kumar**

GitHub:
https://github.com/anubhav7790

## ⭐ Project Highlights

> **Machine Learning + Explainable AI + Streamlit Deployment**

The key objective of this project is not only to predict loan approval but also to make the prediction **interpretable and understandable to users**.
