import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

#load data
df = pd.read_csv('output/data_cleaned.csv')
df = df.dropna()
df = df[df['SEVERITY'] != 4]

X = df.drop(columns=['SEVERITY', 'ACCIDENT_NO'])
y = df['SEVERITY']

# ont hot encoding for low dimension data
onehot_cols = ['FUEL_TYPE', 'VEHICLE_TYPE_DESC', 'VEHICLE_COLOUR_1']

# abelEncoder for high dimension data
label_cols = ['VEHICLE_MAKE', 'VEHICLE_BODY_STYLE']

# aopy DataFrame
df_encoded = df.copy()

# process LabelEncoding
for col in label_cols:
    df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col].astype(str))  # 强制为字符串以防 NaN 报错

# process OneHotEncoding
encoder = OneHotEncoder(sparse_output=False, drop=None)
encoded_array = encoder.fit_transform(df_encoded[onehot_cols])
encoded_cols = encoder.get_feature_names_out(onehot_cols)
encoded_df = pd.DataFrame(encoded_array, columns=encoded_cols)

# merge columns after encoding
df_encoded = pd.concat([df_encoded.drop(columns=onehot_cols).reset_index(drop=True),
                        encoded_df.reset_index(drop=True)], axis=1)

# remove nah
df_encoded = df_encoded.dropna()

X = df_encoded.drop(columns=['SEVERITY', 'ACCIDENT_NO'])
y = df_encoded['SEVERITY']

# Split dataset (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.3, random_state=42
)

# Upsample minority classes (train set only)
train_data = pd.concat([X_train, y_train], axis=1)
max_count = train_data['SEVERITY'].value_counts().max()

upsampled = []
for label in train_data['SEVERITY'].unique():
    subset = train_data[train_data['SEVERITY'] == label]
    upsampled.append(resample(subset,
                              replace=True,
                              n_samples=max_count,
                              random_state=42))
train_upsampled = pd.concat(upsampled)
X_train_up = train_upsampled.drop(columns=['SEVERITY'])
y_train_up = train_upsampled['SEVERITY']

# Feature Selection (train only): remove low-variance features
selector = VarianceThreshold(threshold=0.01)
X_train_up_fs = selector.fit_transform(X_train_up)

# Apply same transformation to test (without fitting again!)
X_test_fs = selector.transform(X_test)

# Feature scaling (train only)
scaler = StandardScaler()
X_train_up_scaled = scaler.fit_transform(X_train_up_fs)
X_test_scaled = scaler.transform(X_test_fs)

# Train logistic regression model
model = LogisticRegression(multi_class='multinomial',
                           solver='lbfgs',
                           class_weight=None,
                           max_iter=1000,
                           random_state=42)
model.fit(X_train_up_scaled, y_train_up)

# Predict & evaluate
y_pred = model.predict(X_test_scaled)

print("\n Classification Report (LogReg, upsampled, clean):")
print(classification_report(y_test, y_pred))
print(f" Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=[1, 2, 3])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[1, 2, 3], yticklabels=[1, 2, 3])
plt.title("Logistic Regression Confusion Matrix (Upsampled + Clean FS)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()


