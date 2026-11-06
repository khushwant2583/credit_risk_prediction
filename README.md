<img width="1911" height="930" alt="image" src="https://github.com/user-attachments/assets/b4d031a3-daae-41f2-841d-9eeeba67e5f6" />

View app at : http://creditriskprediction-vwzidshdx3cmtu3a53euue.streamlit.app/
# AI-Powered Credit Risk & Loan Default Prediction

An end-to-end ML portfolio project to estimate loan default risk, compare baseline and tree-based models, inspect classification metrics, and explain predictions.

## Important data note
The app includes a **synthetic demo dataset generator** so it runs immediately. Synthetic data is useful for demonstrating the workflow, but it is not evidence of real-world credit performance. For a credible portfolio evaluation, replace it with a documented public credit-risk dataset and report its provenance, target definition, class balance, and limitations.

## Run
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

You can also upload a CSV. Select the target column and identify which value means "default". Do not use protected attributes to make lending decisions. This educational demo is not a lending decision system.

## Models and metrics
- Logistic Regression
- Random Forest
- XGBoost
- Precision, recall, F1, ROC-AUC, PR-AUC and confusion matrix
- Optional SHAP global feature importance

## Resume bullet (only after you run and validate it)
"Developed an end-to-end credit-risk classification application comparing Logistic Regression, Random Forest and XGBoost, with stratified evaluation using ROC-AUC/PR-AUC and SHAP-based feature explanations."

Replace this bullet with your measured results; do not claim a score until you have run a suitable real dataset.
