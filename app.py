import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from data_utils import make_demo_data
from modeling import train_models

st.set_page_config(page_title="Credit Risk Lab", page_icon="💳", layout="wide")
st.title("💳 AI-Powered Credit Risk Prediction")
st.caption("Educational ML workflow — not a real lending or eligibility decision tool.")

source=st.radio("Data source", ["Synthetic demo dataset", "Upload CSV"], horizontal=True)
if source == "Synthetic demo dataset":
    df=make_demo_data()
    target="default"
    default_value="1"
else:
    file=st.file_uploader("Upload a labelled CSV", type=["csv"])
    if not file:
        st.info("Upload a CSV containing a binary target column.")
        st.stop()
    df=pd.read_csv(file)
    target=st.selectbox("Target column", df.columns)
    values=df[target].dropna().astype(str).unique().tolist()
    default_value=st.selectbox("Which target value represents default?", values)
    if len(values) != 2:
        st.warning("This starter app expects a binary target with exactly two classes.")

st.subheader("Dataset preview")
st.write(f"{len(df):,} rows · {len(df.columns)} columns")
st.dataframe(df.head(20), use_container_width=True)
if st.button("Train and compare models", type="primary"):
    try:
        with st.spinner("Training models and evaluating on a held-out test set..."):
            results, fitted, X_test, y_test, numeric, categorical = train_models(df, target, default_value)
        st.session_state["results"]=results
        st.session_state["fitted"]=fitted
        st.session_state["X_test"]=X_test
        st.session_state["y_test"]=y_test
        st.session_state["feature_columns"]=list(X_test.columns)
    except Exception as e:
        st.error(f"Training failed: {e}")

if "results" in st.session_state:
    results=st.session_state["results"]
    st.subheader("Model comparison")
    metric_df=pd.DataFrame({name:{k:v for k,v in metrics.items() if k!="Confusion matrix"}
                            for name,metrics in results.items()}).T
    st.dataframe(metric_df.style.format("{:.3f}"), use_container_width=True)
    chosen=st.selectbox("Inspect model", list(results.keys()))
    cm=results[chosen]["Confusion matrix"]
    fig, ax=plt.subplots()
    ax.imshow(cm)
    ax.set_title(f"{chosen} — Confusion Matrix")
    ax.set_xlabel("Predicted label"); ax.set_ylabel("Actual label")
    ax.set_xticks([0,1], ["No default","Default"])
    ax.set_yticks([0,1], ["No default","Default"])
    for (i,j), value in __import__("numpy").ndenumerate(cm):
        ax.text(j,i,str(value),ha="center",va="center")
    st.pyplot(fig); plt.close(fig)
    st.caption("Threshold is fixed at 0.50 in this starter version. Tune it using validation data and business costs before interpreting risk.")
    with st.expander("Explainability (SHAP) — optional"):
        st.write("Install SHAP from requirements, then add a model-specific explainer workflow. Avoid interpreting synthetic-data explanations as real credit policy.")
