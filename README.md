# 🫁 Lung Cancer Detection System
### Machine Learning-Powered Diagnostic Tool for Healthcare Workflows

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Random%20Forest-brightgreen.svg)]()

---

## 📌 Overview
This repository contains an end-to-end Machine Learning web application designed to predict the likelihood of lung cancer. Built with a strict focus on **diagnostic accuracy and model explainability**, this tool leverages a Random Forest classifier (achieving 90% accuracy) deployed via a highly interactive Streamlit dashboard.

The project is designed to simulate how predictive analytics can assist healthcare professionals by providing real-time, data-driven insights to support clinical decision-making and patient screening.

---

## ✨ Key Features
- **High-Accuracy Prediction:** Utilizes a trained Random Forest model to evaluate clinical patient data with 90% accuracy.  
- **Interactive UI:** A clean, responsive Streamlit interface that allows for seamless, user-friendly input of health parameters.  
- **Real-Time Analysis:** Generates instant risk assessments and probability scores based on user inputs.  
- **Transparent AI:** Emphasizes model explainability to ensure that healthcare stakeholders can understand the "why" behind the algorithm's predictions.  

---

## 🛠️ Tech Stack
- **Language:** Python  
- **Machine Learning:** Scikit-Learn (Random Forest)  
- **Frontend/UI:** Streamlit, HTML, CSS  
- **Data Manipulation:** Pandas, NumPy  

---

## 🚀 Installation & Usage

### Prerequisites
Make sure you have Python installed on your machine. It is recommended to use a virtual environment.

### 1. Clone the repository
```bash
git clone https://github.com/Dante-hero/Lung_cancer_pred.git
cd Lung_cancer_pred
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit App
```bash
streamlit run app.py
```

> ⚠️ Note: Replace `app.py` with the actual name of your main Python file if it differs.

---

## 🔮 Future Scope
- Integration of advanced Explainable AI (XAI) tools like SHAP or LIME for deeper insights  
- Expansion of dataset with multimodal data (e.g., X-ray images + clinical data)  
- Docker containerization for scalable deployment on AWS/Azure  

---
