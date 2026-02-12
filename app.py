import streamlit as st
import pandas as pd

st.title("Streamlit: ML App")

model_type = st.selectbox("Choose Model", ["Random Forest", "XGBoost"])

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if model_type:
    st.write(" Model selected ", model_type)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df)