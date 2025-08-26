# 💳 Real-Time Credit Card Fraud Detection System

<img width="1881" height="1030" alt="Image" src="https://github.com/user-attachments/assets/57a5b448-68ea-4ac4-a0d3-1b7d6c75eba2" />

A comprehensive machine learning project that identifies fraudulent credit card transactions in real-time. This system includes a complete pipeline from data preprocessing and feature engineering to model training and deployment with an interactive user interface.

**[Link to Live Demo](https://fraud-detection-system-vivekml.streamlit.app)**

---

## ## 📖 Table of Contents
- [Project Overview](#-project-overview)
- [System Architecture](#-system-architecture)
- [Methodology](#-methodology)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Setup and Usage](#-setup-and-usage)
- [Future Improvements](#-future-improvements)

---

## ## 📜 Project Overview

Credit card fraud is a major concern for financial institutions, leading to significant financial losses. The goal of this project is to develop a robust and efficient system capable of detecting fraudulent transactions with high accuracy.

This project tackles the problem by leveraging a powerful **XGBoost classification model**. The model is trained on a highly imbalanced dataset and is deployed via a **Streamlit** web application, which serves as a dashboard for financial investigators to monitor and act on suspicious activity alerts. The entire application is containerized using **Docker** for portability and ease of deployment.


<img width="1912" height="1040" alt="Image" src="https://github.com/user-attachments/assets/8cdd90fb-7c3b-46c4-aa56-a14067f26b68" />

---

## ## 🏗️ System Architecture

The system follows a standard machine learning project workflow, from data ingestion to user-facing predictions.

1.  **Data Ingestion & Preprocessing:** The raw dataset from Kaggle is loaded and preprocessed. This involves scaling numerical features like `Amount` and `Time` to ensure they are on a comparable scale for the model.
2.  **Feature Engineering:** New features are engineered to provide the model with more predictive signals. In a real-world scenario, this would include transaction frequency, deviation from average transaction amounts, and geo-location anomalies.
3.  **Model Training (Offline):** The preprocessed data is used to train an XGBoost model. A key challenge is the severe class imbalance, which is addressed using the `scale_pos_weight` parameter in the model.
4.  **Model Serialization:** The trained model is saved as a `.pkl` file for later use.
5.  **Streamlit Application:** A user-friendly web interface loads the saved model to make real-time predictions on new transaction data input by the user.
6.  **Docker Containerization:** The entire application, including all dependencies and the trained model, is packaged into a Docker container for consistent and reproducible deployment.

---

## ## 🔬 Methodology

### ### Data
The project uses the **"Credit Card Fraud Detection"** dataset available on [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
- It contains **284,807 transactions**, of which only **492 (0.172%)** are fraudulent.
- Due to confidentiality, the original features have been transformed via Principal Component Analysis (PCA), resulting in 28 anonymized features (`V1` to `V28`).

### ### Modeling: XGBoost
**XGBoost (Extreme Gradient Boosting)** was chosen for this task due to its:
- **High Performance:** It is renowned for its predictive accuracy and speed.
- **Handling of Imbalanced Data:** XGBoost includes a built-in parameter, `scale_pos_weight`, which is specifically designed to handle imbalanced classes. It gives more weight to the minority class (fraudulent transactions), forcing the model to pay more attention to them. This is more efficient than data-level techniques like SMOTE for tree-based models.

### ### Evaluation
Given the class imbalance, **accuracy is a misleading metric**. A model that always predicts "Not Fraud" would achieve over 99% accuracy. Therefore, the following metrics were used for a more meaningful evaluation:
- **Recall:** Measures the model's ability to identify all actual fraudulent transactions. This is the most critical metric, as failing to detect a fraud (a false negative) is very costly.
- **AUPRC (Area Under the Precision-Recall Curve):** AUPRC is a more informative metric than AUC-ROC for highly imbalanced datasets. It summarizes the trade-off between precision (the accuracy of positive predictions) and recall.

---

## ## 🛠️ Tech Stack

- **Data Manipulation & Analysis:** `Pandas`, `NumPy`
- **Machine Learning:** `Scikit-learn`, `XGBoost`
- **Web Framework & UI:** `Streamlit`
- **Containerization:** `Docker`
- **Version Control:** `Git` & `GitHub`

---

## ## 📂 Project Structure
fraud-detection-system/
│
├── .streamlit/
│   └── config.toml           # Streamlit theme configuration
│
├── data/
│   └── creditcard.csv        # The dataset
│
├── saved_model/
│   └── xgboost_model.pkl     # The serialized trained model
│
├── .gitignore                # Files to be ignored by Git
├── app.py                    # The main Streamlit application script
├── Dockerfile                # Instructions for building the Docker image
├── fraud_detection_notebook.ipynb # Jupyter Notebook for EDA and model training
├── README.md                 # Project documentation
└── requirements.txt          # Python dependencies

---

## ## 🚀 Setup and Usage

### ### Local Setup
1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/](https://github.com/)[Your-Username]/[Your-Repo-Name].git
    cd [Your-Repo-Name]
    ```
2.  **Create a Virtual Environment**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the App**
    ```bash
    streamlit run app.py
    ```

### ### Docker
1.  **Build the Docker Image**
    ```bash
    docker build -t fraud-detection-app .
    ```
2.  **Run the Docker Container**
    ```bash
    docker run -p 8501:8501 fraud-detection-app
    ```
    The application will be accessible at `http://localhost:8501`.

---

## ## 📈 Future Improvements

- **REST API:** Deploy the model as a REST API using FastAPI for better integration with other services.
- **Automated Retraining:** Implement a pipeline for automatically retraining the model on new data.
- **Model Monitoring:** Set up a system to monitor model performance and detect concept drift over time.
- **Advanced Features:** Incorporate more complex features, such as user's historical transaction data and time-series analysis.
