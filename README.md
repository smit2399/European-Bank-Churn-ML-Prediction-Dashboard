# 🏦 European Bank Customer Churn Analytics & Prediction

## 🚀 Project Overview

Customer churn is one of the biggest challenges in the banking industry. Retaining existing customers is significantly more cost-effective than acquiring new ones.

This project focuses on **analyzing customer behavior and predicting churn risk** using Machine Learning and interactive data visualization.

👉 The goal is to help banks **identify high-risk customers early and take proactive retention actions**.

---

## 🎯 Business Problem

Banks lose revenue when customers leave (churn). However, not all customers have the same risk level.

Key questions this project answers:
- Which customers are likely to churn?
- What factors drive churn behavior?
- Which customer segments are most at risk?
- How can businesses prioritize retention strategies?

---

## 💡 Solution Approach

This project combines:
- 📊 Exploratory Data Analysis (EDA)
- 🧠 Feature Engineering
- 🤖 Machine Learning (XGBoost)
- 🌐 Interactive Dashboard (Streamlit)

👉 The result is a **real-time churn prediction system + analytics dashboard**

---

## 📊 Key Features of the App

### 📌 1. KPI Dashboard
- Overall churn rate
- Geography-based risk (Germany, Spain, France)
- High balance risk customers
- Inactive customer churn insights

---

### 🌍 2. Churn by Geography
- Identifies high-risk countries
- Helps regional strategy planning

---

### 👥 3. Customer Behavior Analysis
- Age vs Balance visualization
- Detects behavioral churn patterns

---

### 🧠 4. Customer Segmentation
- Segmentation based on financial behavior
- Helps target specific customer groups

---

### 🤖 5. Churn Prediction Engine
- Real-time prediction using ML model
- Takes customer inputs and outputs:
  - Churn probability
  - Risk classification (High / Low)

---

### 📊 6. Feature Importance
- Highlights key drivers of churn
- Helps business decision-making

---

## 🧠 Machine Learning Details

### 🔹 Model Used
- XGBoost Classifier

### 🔹 Problem Type
- Binary Classification (Churn vs No Churn)

---

### 🔹 Feature Engineering

To improve model performance, several derived features were created:

- **Age Groups**
  - Young (≤30)
  - Middle-aged (30–60)
  - Senior (>60)

- **Credit Bands**
  - Low, Medium

- **Balance Segments**
  - Zero Balance
  - Low Balance
  - Medium Balance

- **Customer Segment**
  - High-value vs low-value customers

- **One-Hot Encoding**
  - Geography
  - Gender

---

### 🔹 Key Insight

👉 The model does not rely only on raw data —  
it uses **engineered features that capture customer behavior patterns**

---

## ⚙️ Tech Stack

| Category | Tools |
|--------|------|
| Language | Python |
| Data Processing | Pandas, NumPy |
| Visualization | Plotly |
| ML Model | XGBoost |
| Deployment | Streamlit Cloud |

---

## ▶️ How to Run Locally

### Step 1: Clone the repository

git clone https://github.com/your-username/European-Bank-Churn-ML-Prediction-Dashboard.git
cd European Bank Churn ML Prediction Dashboard

### Step 2: Install dependencies
pip install -r requirements.txt

### Step 3: Run the app
streamlit run app.py

---

## 🌐 Live Demo
👉 (Add your Streamlit app link here after deployment)

---

## 📌 Business Insights

Germany shows highest churn risk
Inactive customers churn significantly more
High balance customers need retention strategies
Age group 45+ shows higher churn probability

---

## 👨‍💻 Author
Smit Prajapati

Aspiring Data Scientist | Data Analyst | ML Engineer

---
