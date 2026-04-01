import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Breast Cancer Predictor", layout="wide")

@st.cache_data
def load_data(path="data.csv"):
    df = pd.read_csv(path)
    df = df.copy()
    if "Unnamed: 32" in df.columns:
        df = df.drop(columns=["Unnamed: 32"])
    if "id" in df.columns:
        df = df.drop(columns=["id"])
    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})
    X = df.drop(columns=["diagnosis"])
    y = df["diagnosis"]
    return X, y

@st.cache_data
def train_model(X, y):
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=2000, solver="liblinear")
    model.fit(X_scaled, y)

    return model, scaler

@st.cache_data
def get_feature_defaults(X):
    return X.mean().to_dict(), X.min().to_dict(), X.max().to_dict()

X, y = load_data()
model, scaler = train_model(X, y)
mean_vals, min_vals, max_vals = get_feature_defaults(X)

st.title("Breast Cancer Detection Web App")
st.markdown("Enter measurable tumor features to get a model prediction (Benign vs Malignant).")

col1, col2 = st.columns(2)
user_input = {}
feature_names = X.columns.tolist()

with col1:
    for f in feature_names[:15]:
        user_input[f] = st.number_input(
            label=f,
            min_value=float(min_vals[f]),
            max_value=float(max_vals[f]),
            value=float(mean_vals[f]),
            format="%.6f",
        )

with col2:
    for f in feature_names[15:]:
        user_input[f] = st.number_input(
            label=f,
            min_value=float(min_vals[f]),
            max_value=float(max_vals[f]),
            value=float(mean_vals[f]),
            format="%.6f",
        )

if st.button("Predict"):
    X_input = pd.DataFrame([user_input])
    X_input_scaled = scaler.transform(X_input)
    prob = model.predict_proba(X_input_scaled)[0, 1]
    pred = model.predict(X_input_scaled)[0]

    label = "Malignant" if pred == 1 else "Benign"
    st.subheader("Prediction result")
    st.write(f"**Prediction:** {label}")
    st.write(f"**Malignant probability:** {prob:.4f}")

    st.info("This model is for educational/demo purposes only and not for clinical use.")

st.markdown("---")

st.subheader("Model performance on full dataset")
accuracy = model.score(scaler.transform(X), y)
st.write(f"Accuracy (train data): {accuracy * 100:.2f}%")

st.write("Use the input fields above and click Predict to classify the tumor as benign or malignant.")
