# 🚗 Vehicle Accident Severity Analysis Project

## 📁 Project Structure and File Descriptions

This project analyzes traffic accident severity using classification, regression, clustering, and rule-based approaches. The project is structured into several task folders, each representing a different phase of the analysis.

### 🧠 Key Functional Files

#### `task_1/`
Initial data inspection, missing value handling, and word cloud visualization:
- `category.py`: create categories for accident severity level
- `cleaning.py`: data cleaning for the original datasets

#### `task_2/`
Preprocessing and encoding logic (file list assumed, not specified).
- `mi.py`: calculate the mutual information for all features vs severity
- `pearson.py`: calculate the pearson correlation for continuous variable vs severity

#### `task_3/`
Core machine learning implementation for classification and rule extraction:
- `baseline.py`: Baseline ZeroR model that always predicts the most frequent class.
- `desicion_tree.py`: Shallow decision tree to extract interpretable fatality risk profiles.
- `random_forest.py`: Random Forest classifier using K-Fold CV with training set upsampling and feature importance visualization.
- `regression.py`: Logistic regression model to assess accident likelihood.
- `task3_guideline`: Guideline or instruction notes (text file).

#### `task_4/`
Unsupervised clustering and risk profiling (implementation assumed).
- `clustering.py`: clustering using KMeans and visualization

---

## ✅ How to Run

If your goal is to run the classification model with upsampling and feature selection:

```bash
cd task_3
python random_forest.py
```

If you want to test rule extraction or baseline:

```bash
python desicion_tree.py  # For rule-based fatality analysis
python baseline.py       # For baseline model
```

Ensure `data_cleaned.csv` is present in the project root directory.

---

## 📝 Notes

- All scripts are commented for clarity.
- Any visualization diagram is visualized per fold and saved to `/output/`.

---

## 📎 Submission Info

This README explains the role of each file and how to execute the main scripts. All required code files are included in this submission. 
