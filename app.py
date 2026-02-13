import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report

import xgboost

models = {
    "Logistic Regression": pickle.load(open("model/logistic_model.pkl", "rb")),
    "Decision Tree": pickle.load(open("model/decision_tree_model.pkl", "rb")),
    "KNN": pickle.load(open("model/KNN_classifier.pkl", "rb")),
    "Naive Bayes":pickle.load(open("model/Naive_bayes_Gaussian_model.pkl", "rb")),
    "Random Forest": pickle.load(open("model/random_forest_model.pkl", "rb")),
    "XGBoost": pickle.load(open("model/XGB_classifier_model.pkl", "rb")),
}

# # Load saved model
# logistic_model = pickle.load(open("logistic_model.pkl", "rb"))
# dt_model = pickle.load(open("dt_model.pkl", "rb"))
# knn_model = pickle.load(open("knn_model.pkl", "rb"))
# naive_bayes = pickle.load(open("knn_model.pkl", "rb"))
# rf_model = pickle.load(open("rf_model.pkl", "rb"))
# xgb_model = pickle.load(open("xgb_model.pkl", "rb"))

st.title("Breast Cancer Prediction")
model_selected = st.selectbox(
    "Select Model",
    ["Logistic Regression", "Decision Tree", "KNN","Naive Bayes", "Random Forest", "XGBoost" ]
)

uploaded_file = st.file_uploader("Upload Test Data in CSV format", type=["csv"])

st.divider()

if model_selected:
    st.write(" Model selected ", model_selected)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df)

selected_metrics = models[model_selected]

st.subheader(selected_metrics['name']+" - Evaluation Metrics")

col1, col2, col3 = st.columns(3)
col1.metric("Accuracy", selected_metrics["accuracy"])
col2.metric("Precision", selected_metrics["precision"])
col3.metric("Recall", selected_metrics["recall"])
col4, col5, col6 = st.columns(3)

col4.metric("F1 Score", selected_metrics["f1_score_result"])
col5.metric("AUC", selected_metrics["auc_score"])
col6.metric("MCC", selected_metrics["mcc_score"])

st.subheader("Classification Report")
report_dict = classification_report(selected_metrics['y_test'], selected_metrics['y_pred'], output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
st.dataframe(report_df)
