import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr 

# load data
df = pd.read_csv("output/data_cleaned.csv")
# === 6. Pearson Correlation (All numeric features vs SEVERITY) ===
pearson_results = []

# === 2. Select continuous numerical features (excluding SEVERITY and ACCIDENT_ID) ===
for col in ['TARE_WEIGHT', 'AGE']:
    try:
        sub_df = df[[col, 'SEVERITY']].replace([np.inf, -np.inf], np.nan).dropna()
        r, p = pearsonr(sub_df[col], sub_df['SEVERITY'])
        pearson_results.append({
            'Feature': col,
            'Pearson_r': r,
            'Abs_r': abs(r),
            'p_value': p
        })
    except Exception as e:
        print(f"Error processing {col}: {e}")

pearson_df = pd.DataFrame(pearson_results).sort_values(by='Abs_r', ascending=False)

# === 7. Print and visualize ===
print("\n Pearson Correlation (vs SEVERITY):")
print(pearson_df[['Feature', 'Pearson_r', 'p_value']].to_string(index=False))





