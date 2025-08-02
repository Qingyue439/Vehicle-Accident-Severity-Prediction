import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
import re
from nltk.corpus import stopwords
import nltk
import seaborn as sns
nltk.download('stopwords')

# Load dataset
df = pd.read_csv('output/vehicle_severity.csv')

# ===== Missing Value Analysis =====
missing_ratio = (df.isnull().sum() / len(df)) * 100
missing_ratio = missing_ratio[missing_ratio > 0].sort_values(ascending=False)

# Plot top N columns with most missing values
top_n = 10
missing_top = missing_ratio.head(top_n)
plt.figure(figsize=(10, 6))
plt.barh(missing_top.index, missing_top.values, color='coral')
plt.xlabel("Missing Ratio (%)")
plt.title(f"Top {top_n} Columns with Most Missing Values (by Ratio)")
plt.gca().invert_yaxis()
plt.tight_layout()
# plt.show()

# Drop columns with more than 70% missing
df.drop(columns=missing_ratio[missing_ratio > 70].index.tolist(), inplace=True)

# ===== Text Cleaning for Object Columns =====
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = text.upper().strip()
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].astype(str).apply(clean_text)

# ===== Drop Unrelated Columns =====
unrelated_cols = ['INITIAL_DIRECTION', 'ROAD_SURFACE_TYPE', 'ROAD_SURFACE_TYPE_DESC', 'REG_STATE', 'FINAL_DIRECTION',
                  'DRIVER_INTENT', 'VEHICLE_MOVEMENT', 'TRAILER_TYPE', 'VEHICLE_COLOUR_2', 'CAUGHT_FIRE',
                  'INITIAL_IMPACT', 'LAMPS', 'TOWED_AWAY_FLAG', 'TRAFFIC_CONTROL', 'TOTAL_NO_OCCUPANTS',
                  'TRAFFIC_CONTROL_DESC', 'VEHICLE_ID', 'VEHICLE_DCA_CODE', 'LEVEL_OF_DAMAGE']
df.drop(columns=[col for col in unrelated_cols if col in df.columns], inplace=True)

# Fill object columns with mode
df.fillna(df.mode().iloc[0], inplace=True)

# ===== FUEL_TYPE Cleaning =====
valid_fuel_types = {'D', 'E', 'G', 'M', 'P', 'R', 'Z'}
df['FUEL_TYPE'] = df['FUEL_TYPE'].astype(str).str.strip().str.upper()
df['FUEL_TYPE'] = df['FUEL_TYPE'].replace('NAH', 'Z')
df.loc[~df['FUEL_TYPE'].isin(valid_fuel_types), 'FUEL_TYPE'] = 'Z'

# ===== VEHICLE_MAKE Cleaning =====
raw_makes = [...]  # List omitted for brevity
valid_makes = set([s.upper().replace(' ', '').replace('/', '').replace('-', '') for s in raw_makes])
df['VEHICLE_MAKE'] = df['VEHICLE_MAKE'].astype(str).str.upper().str.replace(r'[^A-Z0-9]', '', regex=True)
df.loc[~df['VEHICLE_MAKE'].isin(valid_makes), 'VEHICLE_MAKE'] = 'UNKNOWN'

# ===== VEHICLE_BODY_STYLE Cleaning =====
raw_body_styles = [...]  # List omitted for brevity
valid_body_styles = set([s.upper().replace(' ', '').replace('/', '').replace('-', '') for s in raw_body_styles])
df['VEHICLE_BODY_STYLE'] = df['VEHICLE_BODY_STYLE'].astype(str).str.upper().str.replace(r'[^A-Z0-9]', '', regex=True)
df.loc[~df['VEHICLE_BODY_STYLE'].isin(valid_body_styles), 'VEHICLE_BODY_STYLE'] = 'UNKNOWN'

# ===== VEHICLE_COLOUR_1 Cleaning =====
valid_colours = {...}  # Set omitted for brevity
df['VEHICLE_COLOUR_1'] = df['VEHICLE_COLOUR_1'].astype(str).str.strip().str.upper()
df.loc[~df['VEHICLE_COLOUR_1'].isin(valid_colours), 'VEHICLE_COLOUR_1'] = pd.NA

# ===== TARE_WEIGHT Cleaning =====
plt.figure(figsize=(10, 4))
sns.boxplot(data=df, x='TARE_WEIGHT')
plt.title('Boxplot of TARE_WEIGHT')
plt.tight_layout()
# plt.show()
df = df[df['TARE_WEIGHT'] <= 40000]

# ===== NO_OF_WHEELS Cleaning =====
plt.figure(figsize=(8, 5))
sns.boxplot(x=df['NO_OF_WHEELS'], color='skyblue')
plt.title("Boxplot of NO_OF_WHEELS")
plt.xlabel("Number of Wheels")
plt.tight_layout()
plt.savefig("output/boxplot_no_of_wheels.png", dpi=300)
df = df[df['NO_OF_WHEELS'] != df['NO_OF_WHEELS'].max()]

# Convert all float columns to Int64
df[df.select_dtypes(include=['float']).columns] = df.select_dtypes(include=['float']).apply(np.trunc).astype('Int64')

# Drop description columns and single-value columns
df.drop(columns=[col for col in df.columns if col.endswith('_DESC') and col != 'VEHICLE_TYPE_DESC'], inplace=True)
df.drop(columns=['VEHICLE_TYPE'], inplace=True)
df.drop(columns=[col for col in df.columns if df[col].nunique() == 1], inplace=True)
df.drop(columns=['CONSTRUCTION_TYPE', 'VEHICLE_MODEL'], inplace=True)

# Save top 10 missing values summary
missing_info = df.isnull().sum().to_frame(name='Missing Count')
missing_info['Missing Ratio (%)'] = (missing_info['Missing Count'] / len(df)) * 100
missing_info_sorted = missing_info.sort_values(by='Missing Count', ascending=False)
missing_summary = missing_info_sorted.head(10).reset_index()
missing_summary.columns = ['Column Name', 'Missing Count', 'Missing Ratio (%)']
missing_summary.to_excel('top10_missing_summary.xlsx', index=False)

# Fill missing values < 30%
missing_info = df.isnull().sum().to_frame(name='Missing Count')
missing_info['Missing Ratio (%)'] = (missing_info['Missing Count'] / len(df)) * 100
cols_to_fill = missing_info[(missing_info['Missing Count'] > 0) & (missing_info['Missing Ratio (%)'] < 30)].index
for col in cols_to_fill:
    if df[col].dtype == 'object' or df[col].dtype.name == 'category':
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].median())

# ===== Compute AGE =====
df_2 = pd.read_csv('output/vehicle_severity.csv', usecols=['ACCIDENT_DATE'])
df_3 = pd.read_csv('output/vehicle_severity.csv', usecols=['VEHICLE_YEAR_MANUF'])
accident_year = pd.to_datetime(df['ACCIDENT_DATE'], errors='coerce').dt.year
vehicle_year = df['VEHICLE_YEAR_MANUF']
age = accident_year - vehicle_year
valid_idx = age.notna()
df = df[valid_idx].copy()
df['AGE'] = age[valid_idx]
df['AGE'] = age

# Drop date columns and visualize age outliers
df.drop(columns=['ACCIDENT_DATE', 'VEHICLE_YEAR_MANUF'], inplace=True)
plt.figure(figsize=(8, 5))
sns.boxplot(x=df['AGE'], color='coral')
plt.title("Boxplot of AGE")
plt.xlabel("AGE")
plt.tight_layout()
plt.savefig("output/boxplot_age.png", dpi=300)

# Display unique value counts
for col in df.columns:
    print(f"{col}: {df[col].nunique()} unique values")

# Save cleaned data
df.to_csv('output/data_cleaned.csv', index=False)


