# 🚗 Vehicle Accident Severity Prediction

This project investigates how vehicle characteristics influence the severity of road accidents across Victoria, Australia. Using a combination of data cleaning, supervised learning, and unsupervised clustering techniques, we identify high-risk vehicle profiles based on structural attributes.

## 📊 Project Objective

To predict accident severity levels using vehicle characteristics (e.g., body style, tare weight, seating capacity), and to uncover structural vehicle profiles associated with fatal or serious outcomes.

## 🗂️ Dataset

- `filtered_vehicle.csv`: Vehicle characteristics (e.g., make, type, weight, fuel type)
- `accident.csv`: Accident metadata including severity and time
- Merged using vehicle ID and date to form `data_cleaned.csv`

## 🧪 Methodology

### 🔧 Preprocessing
- Removed features with >80% missing values
- Imputed missing categorical fields with mode; numeric with median
- Created derived features (e.g., vehicle age)
- Converted categorical variables via one-hot and ordinal encoding

### 📈 Supervised Learning
- **Random Forest** (best performance): Used for feature importance and classification
- **Logistic Regression**: Compared recall/precision for minority classes
- **Decision Tree**: Interpretable high-risk profile generation

### 📉 Unsupervised Learning
- **K-Means Clustering**: Based on structural features (tare weight, seating capacity, etc.)
- Used PCA for 2D visualization of clusters
- Identified 3 vehicle groups: passenger cars, buses, and industrial vehicles

## 🔍 Key Findings

- **Tare weight** is the strongest predictor of fatal accidents
- **Heavy-duty vehicles** (e.g., prime movers) have the highest fatality rate despite being less frequent
- **Compact passenger vehicles** are involved in most accidents but with lower fatality rate
- **Clustering** helped reveal interpretable high-risk groups based on physical vehicle attributes
