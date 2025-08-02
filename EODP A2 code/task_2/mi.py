import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


# ==========================
# AGE 分组函数
# ==========================
def categorize_vehicle_year(df, show_plot=True):
    df_copy = df.copy()
    Q1 = df_copy["AGE"].quantile(0.25)
    Q3 = df_copy["AGE"].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    # df_filtered = df_copy[(df_copy["AGE"] >= lower) & (df_copy["AGE"] <= upper)]
    # df_filtered = df[df['AGE'] <= 30]

    if show_plot:
        plt.figure(figsize=(10, 5))
        sns.histplot(df["AGE"], bins=20, kde=True, color='darkorange')
        plt.title("Distribution of AGE of Vehicle")
        plt.xlabel("Vehicle AGE")
        plt.ylabel("Number of Accident")
        plt.tight_layout()
        plt.savefig("output/year_cleaned.png")
        # plt.show()

    df_copy["AGE_CATEGORY"] = pd.cut(df_copy["AGE"],
                                     bins=[-1, 5, 10, 15, 20, float("inf")],
                                     labels=[1, 2, 3, 4, 5])
    return df_copy

# ==========================
# TARE_WEIGHT 分组函数
# ==========================
def categorize_tare_weight(df, show_plot=True):
    df_copy = df.copy()
    desc = df_copy["TARE_WEIGHT"].describe()
    Q1 = desc["25%"]
    Q3 = desc["75%"]
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    # df_filtered = df_copy[(df_copy["TARE_WEIGHT"] >= lower) & (df_copy["TARE_WEIGHT"] <= upper)]
    df_filtered = df[df['TARE_WEIGHT'] <= 40000]

    if show_plot:
        plt.figure(figsize=(10, 5))
        sns.histplot(df_filtered["TARE_WEIGHT"], bins=30, kde=True, color='seagreen')
        plt.title("Distribution of Tare Weight of Vehicle")
        plt.xlabel("TARE_WEIGHT (kg)")
        plt.ylabel("Number of Accident")
        plt.tight_layout()
        plt.savefig("output/weight.png")
        # plt.show()

    df_copy["WEIGHT_CATEGORY"] = pd.cut(df_copy["TARE_WEIGHT"],
                                        bins=[0, 1200, 1700, float("inf")],
                                        labels=[1, 2, 3])
    return df_copy

# ==========================

# Mutual Information Functions
# ==========================
def my_entropy(probs):
    return -np.sum(probs * np.log2(probs))

def conditional_entropy(df):
    H_xy = 0
    total = len(df)
    for x_val in df['X'].unique():
        subset = df[df['X'] == x_val]
        px = len(subset) / total
        py_given_x = subset['Y'].value_counts(normalize=True)
        H_xy += px * my_entropy(py_given_x)
    return H_xy

def mutual_info(df):
    px = df['X'].value_counts(normalize=True)
    py = df['Y'].value_counts(normalize=True)
    Hx = my_entropy(px)
    Hy = my_entropy(py)
    H_y_given_x = conditional_entropy(df)
    MI = Hy - H_y_given_x
    NMI = MI / min(Hx, Hy) if min(Hx, Hy) > 0 else 0
    return {'Hx': Hx, 'Hy': Hy, 'H(Y|X)': H_y_given_x, 'MI': MI, 'NMI': NMI}

# === 1. Load and clean data ===
df = pd.read_csv("output/data_cleaned.csv")
df = df[df['SEVERITY'] != 4].dropna().reset_index(drop=True)


# === 3. 简化 VEHICLE_TYPE_DESC 名称 ===
type_desc_map = {
    'CAR': 'CAR', 'STATION WAGON': 'WAGON', 'UTILITY': 'UTILITY',
    'PANEL VAN': 'VAN', 'LIGHT COMMERCIAL VEHICLE RIGID 45 TONNES GVM': 'LIGHT',
    'HEAVY VEHICLE RIGID 45 TONNES': 'HEAVY', 'TAXI': 'TAXI',
    'PRIME MOVER SINGLE TRAILER': 'PRIME 1', 'BUSCOACH': 'BUS',
    'PRIME MOVER ONLY': 'PRIME', 'PRIME MOVER BDOUBLE': 'PRIME 2',
    'MINI BUS913 SEATS': 'MINI BUS', 'OTHER VEHICLE': 'OTHER',
    'PLANT MACHINERY AND AGRICULTURAL EQUIPMENT': 'PLANT',
    'PRIME MOVER BTRIPLE': 'PRIME 3', 'QUAD BIKE': 'BIKE',
    'NOT KNOWN': 'UNKNOWN', 'RIGID TRUCKWEIGHT UNKNOWN': 'RIGID UNKNOWN',
    'PRIME MOVER NO OF TRAILERS UNKNOWN': 'PRIME UNKNOWN',
    'NOT APPLICABLE': 'N/A', 'MOTOR CYCLE': 'MOTORCYCLE', 'PARKED TRAILERS': 'TRAILER'
}
df['VEHICLE_TYPE_DESC'] = df['VEHICLE_TYPE_DESC'].map(type_desc_map).fillna('OTHER')

# === 2. Apply discretization
df = categorize_vehicle_year(df, show_plot=True)
df = categorize_tare_weight(df, show_plot=True)

# === 3. Prepare all categorical features (discretized + original)
categorical_cols = df.select_dtypes(include='number').columns.tolist()
categorical_cols += ['AGE_CATEGORY', 'WEIGHT_CATEGORY']

# 去重、排除无效列
categorical_mi_features = list(set(categorical_cols) - set(['ACCIDENT_NO', 'SEVERITY', 'AGE', 'TARE_WEIGHT']))


# === 4. Compute MI for each discrete feature vs SEVERITY
results = []
target = 'SEVERITY'

for col in categorical_mi_features:
    sub_df = df[[col, target]].dropna().copy()
    sub_df.columns = ['X', 'Y']
    mi_result = mutual_info(sub_df)
    mi_result['Feature'] = col
    results.append(mi_result)

# === 5. Format result table
result_df = pd.DataFrame(results)
result_df = result_df[['Feature', 'Hx', 'Hy', 'H(Y|X)', 'MI', 'NMI']]
result_df = result_df.sort_values(by='MI', ascending=False)

# === 6. Visualize
nmi_df = result_df[['Feature', 'NMI']].sort_values(by='NMI', ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(data=nmi_df, x='NMI', y='Feature', hue='NMI', dodge=False, palette='crest', legend=False)
plt.title("Normalized Mutual Information between Discretized Features and Severity Level")
plt.xlabel("Normalized Mutual Information (NMI)")
plt.ylabel("Discretized Feature")
plt.tight_layout()
plt.savefig("output/discretized_nmi_barplot.png")
# plt.show()

# === 7. Print table
print("\n📊 NMI for Discretized Numeric Features:")
print(result_df.to_string(index=False))


# 选出需要 One-Hot 的 categorical 特征
categorical_cols = df.select_dtypes(include='object').columns.tolist()
drop_cols = ['ACCIDENT_NO', 'SEVERITY_DESC', 'AGE_CATEGORY', 'WEIGHT_CATEGORY']
categorical_cols = [col for col in categorical_cols if col not in drop_cols]

# 目标变量
y = df['SEVERITY']

# Oridinal encode categorical features
encoder = OrdinalEncoder()
X_encoded = encoder.fit_transform(df[categorical_cols])
feature_names = encoder.get_feature_names_out(categorical_cols)

# 计算 Mutual Information
mi_scores = mutual_info_classif(X_encoded, y, discrete_features=True, random_state=42)

# 整理结果为 DataFrame
mi_df = pd.DataFrame({
    'Feature': feature_names,
    'MI': mi_scores
}).sort_values(by='MI', ascending=False)

# 可视化前 15 个
plt.figure(figsize=(10, 6))
sns.barplot(data=mi_df.head(10), x='MI', y='Feature', palette='rocket')
plt.title("Top 15 One-Hot Encoded Features by Mutual Information with Severity")
plt.xlabel("Mutual Information (MI)")
plt.ylabel("One-Hot Feature")
plt.tight_layout()
plt.savefig("output/onehot_mi_barplot.png")
# plt.show()

# 打印表格
print("\n📊 One-Hot Encoded MI Table:")
print(mi_df.head(20).to_string(index=False))


combined_df = pd.concat([
    result_df[['Feature', 'MI']].assign(Source='Numerical'),
    mi_df[['Feature', 'MI']].assign(Source='Catergorical')
], ignore_index=True)

top_features_df = combined_df.sort_values(by='MI', ascending=False).head(10)

plt.figure(figsize=(10, 7))
sns.barplot(data=top_features_df, x='MI', y='Feature', hue='Source')
plt.title("Top 10 Features by Mutual Information (MI) with Severity")
plt.xlabel("Mutual Information (MI)")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("output/combined_mi_barplot.png")
plt.show()
















