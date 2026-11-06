import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, confusion_matrix)
from xgboost import XGBClassifier

def train_models(df, target, default_value):
    data = df.dropna(subset=[target]).copy()
    y = (data[target].astype(str) == str(default_value)).astype(int)
    if y.nunique() < 2:
        raise ValueError("Target must contain both default and non-default examples.")
    X = data.drop(columns=[target])
    if X.shape[1] == 0:
        raise ValueError("Dataset needs at least one feature column.")
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    transformers=[]
    if numeric:
        transformers.append(("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                                               ("scale", StandardScaler())]), numeric))
    if categorical:
        transformers.append(("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                               ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical))
    preprocessor=ColumnTransformer(transformers)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    models={
        "Logistic Regression": LogisticRegression(max_iter=1500, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=250, min_samples_leaf=3,
                                                  class_weight="balanced", random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.05,
                                 subsample=0.85, colsample_bytree=0.85,
                                 eval_metric="logloss", random_state=42)
    }
    results={}
    fitted={}
    for name, model in models.items():
        pipe=Pipeline([("preprocess", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        proba=pipe.predict_proba(X_test)[:,1]
        pred=(proba >= 0.5).astype(int)
        results[name]={
            "Accuracy": accuracy_score(y_test,pred),
            "Precision": precision_score(y_test,pred,zero_division=0),
            "Recall": recall_score(y_test,pred,zero_division=0),
            "F1": f1_score(y_test,pred,zero_division=0),
            "ROC-AUC": roc_auc_score(y_test,proba),
            "PR-AUC": average_precision_score(y_test,proba),
            "Confusion matrix": confusion_matrix(y_test,pred)
        }
        fitted[name]=pipe
    return results, fitted, X_test, y_test, numeric, categorical
