import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.tree import DecisionTreeClassifier, plot_tree, _tree
import matplotlib.pyplot as plt
from graphviz import Digraph
from sklearn.preprocessing import OrdinalEncoder
import warnings

warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('output/data_cleaned.csv')
df = df.dropna()
df = df[df['SEVERITY'] != 4]

# All categorical features to be encoded using OrdinalEncoder
ordinal_cols = ['FUEL_TYPE', 'VEHICLE_TYPE_DESC', 'VEHICLE_COLOUR_1', 'VEHICLE_MAKE', 'VEHICLE_BODY_STYLE']

# Copy DataFrame
df_encoded = df.copy()

# Apply Ordinal Encoding to all categorical columns
df_encoded[ordinal_cols] = OrdinalEncoder().fit_transform(df_encoded[ordinal_cols].astype(str))

# Drop any remaining missing values
df = df_encoded.dropna()

# ========== Train shallow decision tree ==========
print("\n Training shallow decision tree to extract high-risk profiles (SEVERITY=1)...")

X_profile = df.drop(columns=['SEVERITY', 'ACCIDENT_NO'])
y_profile = df['SEVERITY']

profile_tree = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=100,
    class_weight='balanced',
    random_state=42
)
profile_tree.fit(X_profile, y_profile)


plt.figure(figsize=(20, 10))
plot_tree(profile_tree,
          feature_names=X_profile.columns,
          class_names=["Fatal", "Serious", "Other"],
          filled=True,
          rounded=True)
plt.title("Decision Tree - Accident Severity", fontsize=16)
plt.savefig("output/decision_tree_severity_profile.png", dpi=300)
plt.show()
print(" Decision tree trained and visualized. Please check 'decision_tree_fatal_profile.png' for rule-based fatality combinations.\n")

# ========== fatal path ==========
def get_fatal_paths(tree, feature_names, fatal_class=1):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    paths = []

    def recurse(node, path):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            threshold = tree_.threshold[node]
            name = feature_name[node]
            recurse(tree_.children_left[node], path + [(name, "<=", threshold)])
            recurse(tree_.children_right[node], path + [(name, ">", threshold)])
        else:
            pred_class = tree_.value[node][0].argmax()
            if pred_class == fatal_class:
                paths.append(path)

    recurse(0, [])
    return paths

fatal_paths = get_fatal_paths(profile_tree, X_profile.columns.tolist())

