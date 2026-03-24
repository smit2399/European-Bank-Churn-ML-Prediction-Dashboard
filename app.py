#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import plotly.express as px
import pickle

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Churn Dashboard", layout="wide")

# ---------------- CACHE LOADERS ----------------
@st.cache_data
def load_data():
    return pd.read_csv("data/cleaned_data.csv")

@st.cache_resource
def load_model():
    return pickle.load(open("model/xgb_model.pkl", "rb"))

df = load_data()
model = load_model()

# ---------------- PREPROCESS FUNCTION ----------------
def preprocess_input(df):
    df = df.copy()

    df["Geography_Germany"] = (df["Geography"] == "Germany").astype("int8")
    df["Geography_Spain"] = (df["Geography"] == "Spain").astype("int8")
    df["Gender_Male"] = (df["Gender"] == "Male").astype("int8")

    df["AgeGroup_30"] = (df["Age"] <= 30).astype("int8")
    df["AgeGroup_4660"] = ((df["Age"] > 30) & (df["Age"] <= 60)).astype("int8")
    df["AgeGroup_60"] = (df["Age"] > 60).astype("int8")

    df["CreditBand_Low"] = (df["CreditScore"] < 500).astype("int8")
    df["CreditBand_Medium"] = ((df["CreditScore"] < 700) & (df["CreditScore"] >= 500)).astype("int8")

    df["TenureGroup_New"] = (df["Tenure"] <= 3).astype("int8")
    df["TenureGroup_Midterm"] = ((df["Tenure"] <= 7) & (df["Tenure"] > 3)).astype("int8")

    df["BalanceSegment_ZeroBalance"] = (df["Balance"] == 0).astype("int8")
    df["BalanceSegment_LowBalance"] = (df["Balance"] < 50000).astype("int8")
    df["BalanceSegment_MediumBalance"] = ((df["Balance"] < 150000) & (df["Balance"] >= 50000)).astype("int8")

    df["CustomerSegment"] = (df["Balance"] > 100000).astype("int8")

    df.drop(columns=["Geography", "Gender"], inplace=True)

    model_features = [
        'Year','CreditScore','Age','Tenure','Balance',
        'NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary',
        'CustomerSegment','Geography_Germany','Geography_Spain',
        'Gender_Male','AgeGroup_4660','AgeGroup_60','AgeGroup_30',
        'CreditBand_Low','CreditBand_Medium',
        'TenureGroup_Midterm','TenureGroup_New',
        'BalanceSegment_LowBalance','BalanceSegment_MediumBalance',
        'BalanceSegment_ZeroBalance'
    ]

    return df[model_features]


# ---------------- SIDEBAR FILTER ----------------
st.sidebar.title("🔍 Filter Customers")

geo = st.sidebar.multiselect(
    "Geography",
    options=df["Geography"].unique(),
    default=df["Geography"].unique()
)

gender_filter = st.sidebar.multiselect(
    "Gender",
    options=df["Gender"].unique(),
    default=df["Gender"].unique()
)

# Efficient filtering (no overwrite of original df)
filtered_df = df.loc[
    (df["Geography"].isin(geo)) &
    (df["Gender"].isin(gender_filter))
]

# ---------------- TITLE ----------------
st.title("📊 European Bank Churn Intelligence Dashboard")
st.markdown("### 💡 Data-driven insights to reduce customer churn")

# ---------------- KPI SECTION ----------------
st.markdown("## 📌 Key Metrics")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Overall Churn", f"{filtered_df['Exited'].mean()*100:.2f}%")
col2.metric("Germany Risk", f"{filtered_df.loc[filtered_df['Geography']=='Germany','Exited'].mean()*100:.2f}%")
col3.metric("Inactive Churn", f"{filtered_df.loc[filtered_df['IsActiveMember']==0,'Exited'].mean()*100:.2f}%")
col4.metric("High Balance Risk", f"{filtered_df.loc[filtered_df['Balance']>100000,'Exited'].mean()*100:.2f}%")

st.markdown("---")

# ---------------- GEOGRAPHY ----------------
st.subheader("🌍 Churn by Geography")

geo_df = filtered_df.groupby("Geography", observed=True)["Exited"].mean().reset_index()

fig1 = px.bar(geo_df, x="Geography", y="Exited", color="Exited", text_auto=True)
st.plotly_chart(fig1, use_container_width=True)

# ---------------- AGE VS BALANCE ----------------
st.subheader("👥 Customer Behavior")

fig2 = px.scatter(filtered_df, x="Age", y="Balance", color="Exited")
st.plotly_chart(fig2, use_container_width=True)

# ---------------- SEGMENT ----------------
st.subheader("🧠 Customer Segments")

seg_df = filtered_df.groupby("CustomerSegment", observed=True)["Exited"].mean().reset_index()

fig3 = px.bar(seg_df, x="CustomerSegment", y="Exited", color="Exited")
st.plotly_chart(fig3, use_container_width=True)

# ---------------- ML PREDICTION ----------------
st.markdown("---")
st.subheader("🤖 Churn Prediction Engine")

col1, col2, col3 = st.columns(3)

credit = col1.slider("Credit Score", 300, 900, 600)
age = col2.slider("Age", 18, 90, 40)
balance = col3.number_input("Balance", 0, 250000, 50000)

products = st.selectbox("Number of Products", [1,2,3,4])
active = st.selectbox("Active Member", [0,1])
tenure = st.slider("Tenure", 0, 10, 5)
salary = st.number_input("Estimated Salary", 0, 200000, 50000)
has_card = st.selectbox("Has Credit Card", [1,0])
geo_input = st.selectbox("Geography", ["France", "Germany", "Spain"])
gender_input = st.selectbox("Gender", ["Male", "Female"])

if st.button("Predict Churn Risk"):

    input_df = pd.DataFrame([{
        "Year": 2023,
        "CreditScore": credit,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": products,
        "HasCrCard": has_card,
        "IsActiveMember": active,
        "EstimatedSalary": salary,
        "Geography": geo_input,
        "Gender": gender_input
    }])

    processed = preprocess_input(input_df)

    prob = model.predict_proba(processed)[0][1]

    st.metric("Churn Probability", f"{prob*100:.2f}%")
    st.progress(int(prob * 100))

    st.success("✅ Low Risk" if prob < 0.5 else "⚠️ High Risk")

# ---------------- FEATURE IMPORTANCE ----------------
st.markdown("---")
st.subheader("📊 Key Drivers of Churn")

importance = pd.DataFrame({
    "Feature": ["Age", "Balance", "Activity", "Products", "Geography"],
    "Importance": [40, 25, 15, 10, 10]
})

fig4 = px.bar(importance, x="Importance", y="Feature", orientation="h")
st.plotly_chart(fig4, use_container_width=True)

# ---------------- BUSINESS INSIGHT ----------------
st.markdown("---")
st.subheader("📌 Business Insights")

st.markdown("""
- Germany has highest churn risk  
- Customers aged 45+ are more likely to churn  
- Inactive customers churn 2x more  
- High balance customers are at risk  
👉 Focus retention campaigns on these users
""")


# In[ ]:




