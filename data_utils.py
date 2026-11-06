import numpy as np
import pandas as pd

def make_demo_data(n=2500, random_state=42):
    """Synthetic demonstration data; not a real lending dataset."""
    rng = np.random.default_rng(random_state)
    age = rng.integers(21, 70, n)
    income = np.clip(rng.lognormal(10.8, 0.55, n), 8000, 400000)
    loan_amount = np.clip(rng.lognormal(11.0, 0.65, n), 10000, 1500000)
    employment_years = np.clip(rng.normal(6, 4, n), 0, 35)
    debt_to_income = np.clip(rng.beta(2.2, 5, n), 0.01, 0.95)
    credit_history_years = np.clip(rng.normal(7, 5, n), 0, 35)
    purpose = rng.choice(["home", "vehicle", "education", "personal"], n)
    logit = (-3.0 + 3.2*debt_to_income + 0.0000015*loan_amount
             - 0.000006*income - 0.045*credit_history_years
             - 0.025*employment_years + 0.35*(purpose=="personal")
             + 0.2*(purpose=="vehicle"))
    probability = 1/(1+np.exp(-logit))
    default = rng.binomial(1, probability)
    return pd.DataFrame({
        "age": age, "annual_income": income.round(0),
        "loan_amount": loan_amount.round(0),
        "employment_years": employment_years.round(1),
        "debt_to_income": debt_to_income.round(3),
        "credit_history_years": credit_history_years.round(1),
        "loan_purpose": purpose, "default": default
    })
