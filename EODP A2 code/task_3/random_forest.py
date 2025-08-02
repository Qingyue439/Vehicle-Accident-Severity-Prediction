from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.utils import resample
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# ========== Load and clean data ==========
df = pd.read_csv('output/data_cleaned.csv')
df = df.dropna()
df = df[df['SEVERITY'] != 4]  # Remove invalid class

# # OneHot Encoding for low-cardinality categorical features
# onehot_cols = ['FUEL_TYPE', 'VEHICLE_TYPE_DESC', 'VEHICLE_COLOUR_1']

# # Label Encoding for high-cardinality categorical features
# label_cols = ['VEHICLE_MAKE', 'VEHICLE_BODY_STYLE']

# # Copy DataFrame
# df_encoded = df.copy()

# # Apply Label Encoding
# for col in label_cols:
#     df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col].astype(str))

# # Apply OneHot Encoding
# encoder = OneHotEncoder(sparse_output=False, drop=None)
# encoded_array = encoder.fit_transform(df_encoded[onehot_cols])
# encoded_cols = encoder.get_feature_names_out(onehot_cols)
# encoded_df = pd.DataFrame(encoded_array, columns=encoded_cols)

# # Merge one-hot encoded features
# df_encoded = pd.concat([df_encoded.drop(columns=onehot_cols).reset_index(drop=True),
#                         encoded_df.reset_index(drop=True)], axis=1)

# # Drop any remaining missing values
# df_encoded = df_encoded.dropna()


# All categorical features to be encoded using OrdinalEncoder
ordinal_cols = ['FUEL_TYPE', 'VEHICLE_TYPE_DESC', 'VEHICLE_COLOUR_1', 'VEHICLE_MAKE', 'VEHICLE_BODY_STYLE']

# Copy DataFrame
df_encoded = df.copy()

# Apply Ordinal Encoding to all categorical columns
df_encoded[ordinal_cols] = OrdinalEncoder().fit_transform(df_encoded[ordinal_cols].astype(str))

# Drop any remaining missing values
df_encoded = df_encoded.dropna()


# Prepare features and target
X = df_encoded.drop(columns=['SEVERITY', 'ACCIDENT_NO'])
y = df_encoded['SEVERITY']


# ==== 2. K-Fold Evaluation with upsampling inside each fold ====
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
all_preds_k = []
all_true_k = []

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
    X_train_fold = X.iloc[train_idx].copy()
    y_train_fold = y.iloc[train_idx].copy()
    X_test_fold = X.iloc[test_idx].copy()
    y_test_fold = y.iloc[test_idx].copy()

    # 🔁 Upsample only training set inside the fold
    train_df = pd.concat([X_train_fold, y_train_fold], axis=1)
    cls_1 = train_df[train_df['SEVERITY'] == 1]
    cls_2 = train_df[train_df['SEVERITY'] == 2]
    cls_3 = train_df[train_df['SEVERITY'] == 3]
    max_len = max(len(cls_1), len(cls_2), len(cls_3))
    upsampled = [
        resample(cls_1, replace=True, n_samples=max_len, random_state=fold),
        resample(cls_2, replace=True, n_samples=max_len, random_state=fold),
        resample(cls_3, replace=True, n_samples=max_len, random_state=fold)
    ]
    train_upsampled = pd.concat(upsampled).sample(frac=1, random_state=fold).reset_index(drop=True)
    X_train_up = train_upsampled.drop(columns='SEVERITY')
    y_train_up = train_upsampled['SEVERITY']

    # 🔁 Feature selection inside fold
    model_fs = RandomForestClassifier(n_estimators=100, random_state=42)
    model_fs.fit(X_train_up, y_train_up)

    importances = model_fs.feature_importances_
    selected_features = X_train_up.columns[np.argsort(importances)[-10:]]
    


    # ✅ Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_up[selected_features], y_train_up)
    importances_model = model.feature_importances_

    feat_imp_df = pd.DataFrame({
        'Feature': selected_features,
        'Importance': importances_model
    }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(8, 5))
    sns.barplot(data=feat_imp_df, x='Importance', y='Feature', hue='Feature', palette='viridis', legend=False)
    plt.title(f"Fold - Feature Importance on Training Set")
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(f"output/ordinal_feature_importance.png")
    plt.show()



    y_pred_fold = model.predict(X_test_fold[selected_features])

    all_preds_k.extend(y_pred_fold)
    all_true_k.extend(y_test_fold)

    print(f"\n📂 Fold {fold} classification report:")
    print(classification_report(y_test_fold, y_pred_fold))

print("\n✅ Overall K-Fold Classification Report:")
print(classification_report(all_true_k, all_preds_k))
print(f"✅ Overall Accuracy: {accuracy_score(all_true_k, all_preds_k):.4f}")
# ===== 6. Confusion Matrix =====

# 📌 confusion for accident count
cm = confusion_matrix(all_true_k, all_preds_k, labels=[1, 2, 3])

plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['1 = Fatal', '2 = Injury', '3 = Other'],
            yticklabels=['1 = Fatal', '2 = Injury', '3 = Other'],
            cbar_kws={'label': 'Number of Accidents'})
plt.title("Confusion Matrix (Count)", fontsize=14)
plt.xlabel("Predicted Severity Level")
plt.ylabel("Actual Severity Level")
plt.tight_layout()
plt.savefig("output/confusion_matrix_count.png")
plt.show()

# confusion matrix pre recall
cm_norm = confusion_matrix(all_true_k, all_preds_k, labels=[1, 2, 3], normalize='true')

plt.figure(figsize=(7, 6))
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Oranges',
            xticklabels=['1 = Fatal', '2 = Injury', '3 = Other'],
            yticklabels=['1 = Fatal', '2 = Injury', '3 = Other'],
            cbar_kws={'label': 'Recall (%)'})
plt.title("Normalized Confusion Matrix (Per-Class Recall)", fontsize=14)
plt.xlabel("Predicted Severity Level")
plt.ylabel("Actual Severity Level")
plt.tight_layout()
plt.savefig("output/confusion_matrix_normalized.png")
plt.show()

