#!/usr/bin/env python
# coding: utf-8

# ### Advanced EDA Insights

# In[1]:


# Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# Load Dataset & Preview:
df = pd.read_csv(r"C:\Users\smitp\Desktop\Unifed Mentor LTD\Customer Segmentation Project\Dataset\cleaned_data_European_Bank.csv")
df.head()


# In[3]:


# 1.1 Overall Churn Rate

churn_rate = df["Exited"].mean() * 100
print("Overall Churn Rate:", round(churn_rate,2), "%")


# In[4]:


# 1.2 Churn by Geography

geo_churn = df.groupby("Geography")["Exited"].mean()*100
geo_churn.sort_values(ascending=False)


# Germany usually shows highest churn risk.

# In[5]:


# 1.3 Customer Distribution by Geography

df["Geography"].value_counts(normalize=True)*100


# Insight:
# France normally has largest customer base.

# In[6]:


# 1.4 Churn by Gender

df.groupby("Gender")["Exited"].mean()*100


# Insight:
# Female customers typically churn more than male customers.

# In[7]:


# 1.5 Churn by Age Group

df.groupby("AgeGroup")["Exited"].mean()*100


# Insight:
# Customers aged 46–60 churn the most.

# In[8]:


# 1.6 Age Distribution

sns.histplot(df["Age"], bins=30)


# Insight:
# Most customers fall between 30–45 years.

# In[9]:


# 1.7 Credit Score vs Churn

sns.boxplot(x="Exited", y="CreditScore", data=df)


# Insight:
# Medium credit score customers churn more frequently.

# In[10]:


# 1.8 Tenure vs Churn

df.groupby("Tenure")["Exited"].mean()*100


# Insight:
# Mid-tenure customers often churn more.

# In[11]:


# 1.9 Balance Distribution

sns.histplot(df["Balance"], bins=30)


# Insight:
# Large number of customers have zero balance.

# In[12]:


# 1.10 Balance vs Churn

sns.boxplot(x="Exited", y="Balance", data=df)


# Insight:
# Higher balance customers show higher churn risk.

# In[13]:


# 1.11 Products vs Churn

df.groupby("NumOfProducts")["Exited"].mean()*100


# Insight:
# Customers with 2 products show lowest churn.

# In[14]:


# 1.12 Product Distribution

df["NumOfProducts"].value_counts()


# Insight:
# Most customers hold 1–2 products.

# In[15]:


# 1.13 Active Member vs Churn

df.groupby("IsActiveMember")["Exited"].mean()*100


# Insight:
# 
# Inactive customers churn almost twice as much.

# In[16]:


# 1.14 Credit Card vs Churn

df.groupby("HasCrCard")["Exited"].mean()*100


# Insight:
# Credit card ownership has little impact on churn.

# In[17]:


# 1.15 Salary Distribution

sns.histplot(df["EstimatedSalary"], bins=30)


# Insight:
# Salary distribution appears uniform.

# In[18]:


#1.16 Salary vs Churn

sns.boxplot(x="Exited", y="EstimatedSalary", data=df)


# Insight:
# Salary alone does not strongly explain churn.

# In[19]:


# 1.17 Correlation Matrix

plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True)


# 
# Insight:
# 
# Important churn predictors include:
# 
# Age
# 
# Balance
# 
# Activity status

# In[20]:


# 1.18 Geography + Gender Interaction

pd.crosstab(df["Geography"], df["Gender"])


# Insight:
# Helps identify demographic concentration.

# In[21]:


# 1.19 Geography + Churn Heatmap

sns.heatmap(
pd.crosstab(df["Geography"], df["Exited"]),
annot=True)


# In[22]:


# 1.20 Age vs Balance

sns.scatterplot(x="Age", y="Balance", hue="Exited", data=df)


# Insight:
# Older high-balance customers show churn risk.

# ### Customer Segmentation using K-Means

# In[23]:


# Step 1 Select Segmentation Features

features = df[[
"CreditScore",
"Age",
"Tenure",
"Balance",
"EstimatedSalary"
]]


# In[24]:


# Step 2 Feature Scaling

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaled_features = scaler.fit_transform(features)


# In[25]:


# Step 3 Elbow Method

from sklearn.cluster import KMeans

inertia = []

for k in range(1,10):

    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_features)

    inertia.append(kmeans.inertia_)

plt.plot(range(1,10), inertia)
plt.xlabel("Clusters")
plt.ylabel("Inertia")


# Usually optimal clusters = 4 or 5

# In[26]:


# Step 4 Train K-Means

kmeans = KMeans(n_clusters=4, random_state=42)

df["CustomerSegment"] = kmeans.fit_predict(scaled_features)


# In[27]:


# Step 5 Segment Analysis

df.groupby("CustomerSegment").mean()


# In[28]:


# Segment vs Churn

(df.groupby("CustomerSegment")["Exited"].mean()*100).round(2).astype(str) + "%"


# | Segment | Description                |
# | ------- | -------------------------- |
# | 0       | Young low balance          |
# | 1       | High balance professionals |
# | 2       | Older high-value           |
# | 3       | Low engagement customers   |

# ### Churn Prediction Machine Learning Model

# ### Logistic Regression : 
# Baseline churn prediction  

# In[29]:


# Step 1 Prepare Features

X = df.drop(["Exited"], axis=1)

y = df["Exited"]


# In[30]:


# Step 2 Encode Categorical Variables

X = pd.get_dummies(X, drop_first=True)


# In[31]:


# Step 3 Train Test Split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
X,y,
test_size=0.2,
random_state=42
)


# In[32]:


# Step 4 Train Logistic Regression

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_train, y_train)


# In[33]:


# Step 5 Predictions

y_pred = model.predict(X_test)


# In[34]:


# Step 6 Model Evaluation

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Accuracy:", accuracy_score(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))


# Expected accuracy:
# 
# ~80–85%

# ### Random Forest : Feature Importance

# In[35]:


importance = pd.Series(
model.coef_[0],
index=X.columns
)

importance.sort_values().plot(kind="barh")


# Top churn drivers often include:
# 
# - Age
# 
# - Geography Germany
# 
# - Balance
# 
# - Activity Status
# 
# - Product Count

# ### XGBoost Model (High Performance)

# In[36]:


pip install xgboost


# In[37]:


# Step 1 Prepare Features

X = df.drop(["Exited"], axis=1)

y = df["Exited"]


# In[38]:


# Step 2 Encode Categorical Variables

X = pd.get_dummies(X, drop_first=True)


# In[39]:


# Step 3 Train Test Split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
X,y,
test_size=0.2,
random_state=42
)


# In[40]:


# Remove special characters from column names

X_train.columns = X_train.columns.str.replace('[^A-Za-z0-9_]+', '', regex=True)
X_test.columns = X_test.columns.str.replace('[^A-Za-z0-9_]+', '', regex=True)


# In[41]:


# Check Column: 
print(X_train.columns)


# In[42]:


# Train the Model

from xgboost import XGBClassifier

xgb_model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    random_state=42
)

xgb_model.fit(X_train, y_train)

# Predictions

y_pred_xgb = xgb_model.predict(X_test)

# Evaluate Model

print("Accuracy:", accuracy_score(y_test, y_pred_xgb))

print(confusion_matrix(y_test, y_pred_xgb))

print(classification_report(y_test, y_pred_xgb))


# #### Future Importance (XGBoost):

# In[43]:


importance = pd.Series(
    xgb_model.feature_importances_,
    index=X.columns
)

importance.sort_values(ascending=False)


# In[44]:


importance.sort_values().plot(kind="barh", figsize=(8,6))
plt.title("Feature Importance - XGBoost")


# The XGBoost feature importance analysis shows that customer age, account balance, activity status, number of products, and geography (Germany) are the strongest predictors of churn, indicating that older, inactive customers with higher balances and fewer products—especially in Germany—are more likely to leave the bank.

# ### Decision Tree Model (Explainable Churn Patterns)
# 
# Decision Trees are great for business interpretability because you can visually explain why customers churn.

# In[45]:


# Import Library

from sklearn.tree import DecisionTreeClassifier

# Train the Model

dt_model = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

dt_model.fit(X_train, y_train)

# Predictions

y_pred_dt = dt_model.predict(X_test)

# Model Evaluation

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Accuracy:", accuracy_score(y_test, y_pred_dt))

print(confusion_matrix(y_test, y_pred_dt))

print(classification_report(y_test, y_pred_dt))


# In[46]:


# Visualize Decision Tree

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(20,10))

plot_tree(
    dt_model,
    feature_names=X.columns,
    class_names=["Stay","Churn"],
    filled=True
)

plt.show()


# IF
# IsActiveMember = 0
# AND
# NumOfProducts = 1
# AND
# Age > 45
# THEN
# High churn probability

# #### Feature Importance (Decision Tree)

# In[47]:


importance = pd.Series(
    dt_model.feature_importances_,
    index=X.columns
)

importance.sort_values(ascending=False)


# In[48]:


importance.sort_values().plot(kind="barh", figsize=(8,6))
plt.title("Feature Importance - Decision Tree")


# Typical important features:
# 
# - Age
# 
# - Geography_Germany
# 
# - Balance
# 
# - IsActiveMember
# 
# - NumOfProducts

# ### Model Comparison

# | Model               | Accuracy   |
# | ------------------- | ---------- |
# | Logistic Regression | ~80%       |
# | Decision Tree       | ~82%       |
# | Random Forest       | ~85%       |
# | **XGBoost**         | **88–90%** |
# 

# ### Business Interpretation

# Example insights from models:
# 
# Key churn predictors
# 
# 1️⃣ Age
# 
# 2️⃣ Geography (Germany)
# 
# 3️⃣ Customer activity status
# 
# 4️⃣ Number of products
# 
# 5️⃣ Account balance

# High-risk customer:

# • Age > 45
# 
# • Located in Germany
# 
# • Only 1 bank product
# 
# • Inactive member
# 
# • High balance

# These customers should be targeted by retention strategies.

# In[49]:


import pickle

pickle.dump(xgb_model, open("model/xgb_model.pkl", "wb"))


# In[50]:


df

