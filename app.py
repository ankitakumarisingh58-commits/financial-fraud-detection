import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import os

st.set_page_config(
    page_title="Financial Fraud Detection",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Financial Fraud Detection Dashboard")
st.markdown("### Financial Transaction Monitoring & Fraud Analytics")

# Load Dataset
@st.cache_data
def load_data():
    df = pd.read_csv("synthetic_fraud_dataset1.csv")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df

df = load_data()

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["Dashboard Overview", "Fraud Analysis",
     "Transaction Analysis", "ML Prediction", "Data Explorer"]
)

# Overview
if page == "Dashboard Overview":

    total = len(df)
    fraud = int(df["Fraud_Label"].sum())
    genuine = total - fraud
    percentage = (fraud / total) * 100 if total else 0

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Transactions", f"{total:,}")
    c2.metric("Fraud Transactions", f"{fraud:,}")
    c3.metric("Genuine Transactions", f"{genuine:,}")
    c4.metric("Fraud Rate", f"{percentage:.2f}%")

    st.subheader("Transaction Overview")

    col1, col2 = st.columns(2)

    with col1:
        counts = df["Fraud_Label"].value_counts().rename(
            index={0: "Genuine", 1: "Fraud"}
        )
        fig = px.pie(
            values=counts.values,
            names=counts.index,
            title="Fraud vs Genuine Transactions",
            hole=0.4
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(
            df,
            x="Transaction_Amount",
            color="Fraud_Label",
            title="Transaction Amount Distribution",
            barmode="overlay"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Recent Transactions")
    st.dataframe(df.head(10), use_container_width=True)


# Fraud Analysis
elif page == "Fraud Analysis":

    st.header("Fraud Analysis")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            df.groupby("Transaction_Type")["Fraud_Label"].sum()
            .reset_index(),
            x="Transaction_Type",
            y="Fraud_Label",
            title="Fraud by Transaction Type"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            df.groupby("Location")["Fraud_Label"].sum()
            .reset_index(),
            x="Location",
            y="Fraud_Label",
            title="Fraud by Location"
        )
        st.plotly_chart(fig, use_container_width=True)

    fig = px.bar(
        df.groupby("Device_Type")["Fraud_Label"].sum()
        .reset_index(),
        x="Device_Type",
        y="Fraud_Label",
        title="Fraud by Device Type"
    )
    st.plotly_chart(fig, use_container_width=True)


# Transaction Analysis
elif page == "Transaction Analysis":

    st.header("Transaction Analysis")

    fig = px.histogram(
        df,
        x="Transaction_Amount",
        color="Transaction_Type",
        title="Transaction Amount Analysis"
    )
    st.plotly_chart(fig, use_container_width=True)

    fig = px.bar(
        df.groupby("Merchant_Category")["Transaction_Amount"]
        .sum().reset_index(),
        x="Merchant_Category",
        y="Transaction_Amount",
        title="Merchant Category Analysis"
    )
    st.plotly_chart(fig, use_container_width=True)

    fig = px.box(
        df,
        x="Card_Type",
        y="Transaction_Amount",
        title="Card Type Analysis"
    )
    st.plotly_chart(fig, use_container_width=True)


# Machine Learning Prediction
elif page == "ML Prediction":

    st.header("🤖 Fraud Detection Prediction")

    if not os.path.exists("fraud_model.pkl"):
        st.warning("Please run train_model.py first.")
    else:
        model = joblib.load("fraud_model.pkl")

        with st.form("prediction_form"):

            amount = st.number_input(
                "Transaction Amount", min_value=0.0
            )
            balance = st.number_input(
                "Account Balance", min_value=0.0
            )

            transaction = st.selectbox(
                "Transaction Type", df["Transaction_Type"].unique()
            )
            device = st.selectbox(
                "Device Type", df["Device_Type"].unique()
            )
            location = st.selectbox(
                "Location", df["Location"].unique()
            )
            merchant = st.selectbox(
                "Merchant Category", df["Merchant_Category"].unique()
            )
            card = st.selectbox(
                "Card Type", df["Card_Type"].unique()
            )

            previous = st.selectbox(
                "Previous Fraudulent Activity", [0, 1]
            )
            daily = st.number_input(
                "Daily Transaction Count", min_value=0, value=1
            )
            age = st.number_input(
                "Card Age (Days)", min_value=0, value=100
            )

            date = st.date_input("Transaction Date")

            submit = st.form_submit_button("Predict Fraud")

        if submit:

            record = pd.DataFrame([{
                "Transaction_Amount": amount,
                "Transaction_Type": transaction,
                "Date": pd.Timestamp(date),
                "Account_Balance": balance,
                "Device_Type": device,
                "Location": location,
                "Merchant_Category": merchant,
                "Previous_Fraudulent_Activity": previous,
                "Daily_Transaction_Count": daily,
                "Card_Type": card,
                "Card_Age": age
            }])

            prediction = model.predict(record)[0]
            probability = model.predict_proba(record)[0][1]

            if prediction == 1:
                st.error("⚠️ Fraudulent Transaction Detected!")
            else:
                st.success("✅ Genuine Transaction")

            st.metric("Fraud Probability", f"{probability*100:.2f}%")


# Data Explorer
elif page == "Data Explorer":

    st.header("📊 Dataset Explorer")

    status = st.selectbox(
        "Select Transaction Status",
        ["All", "Fraud", "Genuine"]
    )

    filtered = df.copy()

    if status == "Fraud":
        filtered = filtered[filtered["Fraud_Label"] == 1]
    elif status == "Genuine":
        filtered = filtered[filtered["Fraud_Label"] == 0]

    st.write("Total Records:", len(filtered))

    st.dataframe(filtered, use_container_width=True)

    csv = filtered.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Dataset",
        csv,
        "fraud_data.csv",
        "text/csv"
    )
