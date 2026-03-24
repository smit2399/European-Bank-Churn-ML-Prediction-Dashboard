#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

def preprocess_input(data):
    df = data.copy()

    # Geography Encoding
    df["Geography_Germany"] = (df["Geography"] == "Germany").astype(int)
    df["Geography_Spain"] = (df["Geography"] == "Spain").astype(int)

    # Gender Encoding
    df["Gender_Male"] = (df["Gender"] == "Male").astype(int)

    # Age Groups
    df["AgeGroup_30"] = (df["Age"] <= 30).astype(int)
    df["AgeGroup_4660"] = ((df["Age"] > 30) & (df["Age"] <= 60)).astype(int)
    df["AgeGroup_60"] = (df["Age"] > 60).astype(int)

    # Credit Bands
    df["CreditBand_Low"] = (df["CreditScore"] < 500).astype(int)
    df["CreditBand_Medium"] = ((df["CreditScore"] >= 500) & (df["CreditScore"] < 700)).astype(int)

    # Tenure Groups
    df["TenureGroup_New"] = (df["Tenure"] <= 3).astype(int)
    df["TenureGroup_Midterm"] = ((df["Tenure"] > 3) & (df["Tenure"] <= 7)).astype(int)

    # Balance Segments
    df["BalanceSegment_ZeroBalance"] = (df["Balance"] == 0).astype(int)
    df["BalanceSegment_LowBalance"] = (df["Balance"] < 50000).astype(int)
    df["BalanceSegment_MediumBalance"] = ((df["Balance"] >= 50000) & (df["Balance"] < 150000)).astype(int)

    # Customer Segment
    df["CustomerSegment"] = (df["Balance"] > 100000).astype(int)

    # Drop original categorical columns
    df = df.drop(columns=["Geography", "Gender"])

    # Final feature order (CRITICAL)
    model_features = [
        'Year', 'CreditScore', 'Age', 'Tenure', 'Balance',
        'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
        'CustomerSegment', 'Geography_Germany', 'Geography_Spain',
        'Gender_Male', 'AgeGroup_4660', 'AgeGroup_60', 'AgeGroup_30',
        'CreditBand_Low', 'CreditBand_Medium',
        'TenureGroup_Midterm', 'TenureGroup_New',
        'BalanceSegment_LowBalance', 'BalanceSegment_MediumBalance',
        'BalanceSegment_ZeroBalance'
    ]

    return df[model_features]

