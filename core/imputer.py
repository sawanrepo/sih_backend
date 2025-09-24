import pandas as pd
import numpy as np
import joblib

imputer = joblib.load("ensemble_imputer.pkl") #kept outside fxn for fast response.

def predict_missing(user_row: dict):
    """
    Predict missing numeric values in a single user row using the trained IterativeImputer.
    Works with one-hot encoded 'product_type' and transformed 'weight_kg' and 'energy_mix_pct_renewables'.
    """

    

    user_df = pd.DataFrame([user_row])

    product_type_cols = [col for col in imputer.feature_names_in_ if col.startswith("product_type_")]
    for col in product_type_cols:
        cat_value = col.replace("product_type_", "")
        user_df[col] = 1 if user_row.get("product_type") == cat_value else 0

    if "product_type" in user_df:
        user_df = user_df.drop(columns=["product_type"])

    if 'weight_kg' in user_df:

        user_df['weight_kg_log'] = np.log1p(user_df['weight_kg'].fillna(0))
        user_df.loc[user_df['weight_kg'].isna(), 'weight_kg_log'] = np.nan

    if 'energy_mix_pct_renewables' in user_df:
        user_df['energy_mix_pct_renewables_sqrt'] = np.sqrt(user_df['energy_mix_pct_renewables'].fillna(0))
        user_df.loc[user_df['energy_mix_pct_renewables'].isna(), 'energy_mix_pct_renewables_sqrt'] = np.nan


    for col in imputer.feature_names_in_:
        if col not in user_df.columns:
            user_df[col] = 0


    user_df = user_df[imputer.feature_names_in_]

    imputed_values = imputer.transform(user_df)

    imputed_df = pd.DataFrame(imputed_values, columns=user_df.columns)

    if 'weight_kg_log' in imputed_df:
        imputed_df['weight_kg'] = np.expm1(imputed_df['weight_kg_log'])
    if 'energy_mix_pct_renewables_sqrt' in imputed_df:
        imputed_df['energy_mix_pct_renewables'] = imputed_df['energy_mix_pct_renewables_sqrt'] ** 2

    numeric_cols = ["weight_kg", "energy_mix_pct_renewables", "co2_kg_per_kg",
                    "recycled_content_pct", "lifetime_years", "reuse_probability_pct"]
    return imputed_df[numeric_cols].iloc[0].to_dict()

if __name__ == "__main__":
    user_input = {
          "product_type": "Window Frame",
        "weight_kg": 35,
        "energy_mix_pct_renewables": 50,
        "co2_kg_per_kg": np.nan,           # missing
        "recycled_content_pct": np.nan,    # missing
        "lifetime_years": 25,
        "reuse_probability_pct": 60
    }
    completed = predict_missing(user_input)
    print("Predicted user row with imputed values:\n", completed)