import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, zero_one_loss
from sklearn.model_selection import train_test_split
import warnings
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

warnings.filterwarnings('ignore')

# ========== Load and clean data ==========
df = pd.read_csv('output/data_cleaned.csv')
df = df.dropna()
df = df[df['SEVERITY'] != 4]  # Remove invalid class

# one hot encoding for low dimension variable
onehot_cols = ['FUEL_TYPE', 'VEHICLE_TYPE_DESC', 'VEHICLE_COLOUR_1']

# Label Encoding for high dimension variable
label_cols = ['VEHICLE_MAKE', 'VEHICLE_BODY_STYLE']

# copy DataFrame
df_encoded = df.copy()

#  Label Encoding
for col in label_cols:
    df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col].astype(str))  # 强制为字符串以防 NaN 报错

# OneHot Encoding
encoder = OneHotEncoder(sparse_output=False, drop=None)
encoded_array = encoder.fit_transform(df_encoded[onehot_cols])
encoded_cols = encoder.get_feature_names_out(onehot_cols)
encoded_df = pd.DataFrame(encoded_array, columns=encoded_cols)

# merge the column after 
df_encoded = pd.concat([df_encoded.drop(columns=onehot_cols).reset_index(drop=True),
                        encoded_df.reset_index(drop=True)], axis=1)

# remove missing values
df_encoded = df_encoded.dropna()

X = df_encoded.drop(columns=['SEVERITY', 'ACCIDENT_NO'])
y = df_encoded['SEVERITY']


# ========== Train/test split (no upsampling) ==========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# ========== ZeroR Model ==========
# Predict the most frequent class in training set
most_frequent_class = y_train.mode()[0]
y_pred = np.full_like(y_test, fill_value=most_frequent_class)

# ========== Evaluation ==========
acc = accuracy_score(y_test, y_pred)
z1_loss = zero_one_loss(y_test, y_pred)

print(f"Accuracy (ZeroR Baseline): {acc:.4f}")
print(f"Zero-One Loss: {z1_loss:.4f}")
print("\n Classification Report:")
print(classification_report(y_test, y_pred))


