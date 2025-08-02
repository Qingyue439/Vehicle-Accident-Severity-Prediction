import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
# read data
df = pd.read_csv('C:\\Users\\15500\\Desktop\\data_cleaned (1).csv')
df = df[df['SEVERITY'] != 4].dropna().reset_index(drop=True)


# Define grouping keys and numeric features
group_cols = ['AGE','VEHICLE_MAKE','VEHICLE_BODY_STYLE','FUEL_TYPE','VEHICLE_TYPE_DESC','VEHICLE_COLOUR_1']
feature_cols = ['NO_OF_WHEELS', 'SEATING_CAPACITY', 'TARE_WEIGHT','NO_OF_CYLINDERS']

# Aggregate numeric features by group
agg_df = df.groupby(group_cols)[feature_cols].mean().reset_index()


# Normalize numeric features
features_to_scale = feature_cols
X_scaled = MinMaxScaler().fit_transform(agg_df[features_to_scale])

# Elbow Method for optimal k
inertias = []
K_range = range(1, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K_range, inertias, marker='o', color='blue')
plt.xticks(K_range)
plt.title("Elbow Method for Optimal k")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("WCSS (Inertia)")
plt.grid(True)
plt.tight_layout()
plt.show()

# KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X_scaled)    

# PCA
pca = PCA(n_components=0.8)
X_pca = pca.fit_transform(X_scaled)
agg_df['cluster'] = labels    
agg_df['pca1'] = X_pca[:, 0]
agg_df['pca2'] = X_pca[:, 1]


# Add to raw DataFrame
agg_df['structure_index'] = X_pca[:, 0]

# Merge back cluster to original df
df_clustered = pd.merge(df, agg_df[group_cols + ['cluster']], on=group_cols, how='left')

# Risk analysis
group_summary = df_clustered.groupby('cluster').agg({
    'SEVERITY': ['count', 'mean', lambda x: (x == 1).mean()]
}).reset_index()
group_summary.columns = ['Cluster', 'Total_Accidents', 'Avg_Severity', 'Fatal_Rate']

# Sort clusters by Avg_Severity 
sort_key = 'Avg_Severity'  
severity_order = group_summary.sort_values(by=sort_key, ascending=False).reset_index(drop=True)
severity_order['Sorted_Cluster'] = severity_order.index

# 构建映射字典：旧编号 → 新编号
cluster_map = dict(zip(severity_order['Cluster'], severity_order['Sorted_Cluster']))

# 替换所有表中的 cluster 编号
group_summary['Cluster'] = group_summary['Cluster'].map(cluster_map)
agg_df['cluster'] = agg_df['cluster'].map(cluster_map)
df_clustered['cluster'] = df_clustered['cluster'].map(cluster_map)

# ✅ 最后再按新 Cluster 排序一次（便于画图）
group_summary = group_summary.sort_values('Cluster').reset_index(drop=True)
agg_df = agg_df.sort_values('cluster').reset_index(drop=True)
df_clustered = df_clustered.sort_values('cluster').reset_index(drop=True)


# ✅ 9. Risk profile plot
fig, ax1 = plt.subplots(figsize=(10, 8))
ax1.bar(group_summary['Cluster'], group_summary['Total_Accidents'], color='skyblue', alpha=0.7)
ax1.set_ylabel("Number of Accidents", color='blue')
ax1.set_xlabel("Cluster")

ax2 = ax1.twinx()
ax2.plot(group_summary['Cluster'], group_summary['Avg_Severity'], color='red', marker='o', label='Avg Severity')
ax2.plot(group_summary['Cluster'], group_summary['Fatal_Rate'] * 100, color='black', linestyle='--', marker='x', label='Fatal %')
ax2.set_ylabel("Avg Severity / Fatal %", color='red')

# ─── 给柱子写数量标签 ─────────────────────────────────────────────
for bar, value in zip(ax1.patches, group_summary['Total_Accidents']):
    ax1.annotate(f'{value:,}',                   
                 xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                 xytext=(0, 3),                  
                 textcoords="offset points",
                 ha='center', va='bottom', fontsize=9, color='blue')

# ─── 给折线上写 Avg Severity & Fatal % ───────────────────────────
for x, y in zip(group_summary['Cluster'], group_summary['Avg_Severity']):
    ax2.annotate(f'{y:.2f}',
                 xy=(x, y),
                 xytext=(0, 5),
                 textcoords="offset points",
                 ha='center', va='bottom', fontsize=9, color='red')

for x, y in zip(group_summary['Cluster'], group_summary['Fatal_Rate'] * 100):
    ax2.annotate(f'{y:.1f}%',                   
                 xy=(x, y),
                 xytext=(0, -12),               
                 textcoords="offset points",
                 ha='center', va='top', fontsize=9, color='black')


fig.legend(loc='upper right', bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
plt.title("Risk Profile by Structure Cluster")
plt.tight_layout()
plt.savefig('output/answer_pca.png')
plt.show()

# ✅ 10. PCA scatter plot of clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(data=agg_df, x='pca1', y='pca2', hue='cluster', palette='Set2', edgecolor='black')
plt.title("PCA Scatter of Vehicle Structure Clusters")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.tight_layout()
plt.savefig('output/cluster_pca.png')
plt.show()

heatmap_data = group_summary[['Avg_Severity', 'Fatal_Rate']].copy()
heatmap_data['Fatal_Rate'] = heatmap_data['Fatal_Rate'] 
heatmap_data.index = group_summary['Cluster']  

# ✅ 绘制热力图
plt.figure(figsize=(6, 4))
sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap='YlOrRd', cbar_kws={'label': '% / Severity'})
plt.title("Heatmap of Avg Severity & Fatal Rate by Cluster")
plt.xlabel("Metrics")
plt.ylabel("Cluster")
plt.tight_layout()
plt.savefig("output/cluster_risk_heatmap.png", dpi=300)
plt.show()


# ✅ 11. Cluster-wise summary (mode for categorical + mean for numerical)
num_con_features = ['TARE_WEIGHT', 'AGE']
num_dis_features = ['SEATING_CAPACITY','NO_OF_CYLINDERS','NO_OF_WHEELS']
cat_features = ['VEHICLE_MAKE','VEHICLE_BODY_STYLE','VEHICLE_COLOUR_1','FUEL_TYPE','VEHICLE_TYPE_DESC']

# 平均数（数值特征）
mean_summary = df_clustered.groupby('cluster')[num_con_features].mean().round(2)
median_summary = df_clustered.groupby('cluster')[num_dis_features].median().round(0)

# 众数（类别特征）
mode_summary = df_clustered.groupby('cluster')[cat_features].agg(lambda x: x.mode().iloc[0])

# 合并展示
cluster_profile = pd.concat([mean_summary, mode_summary, median_summary], axis=1)

print("\n📊 Cluster-wise Profile (Mean + Mode):")
print(cluster_profile)
cluster_profile.to_csv('output/cluster_profile.csv', index=True)


# ✅ 1. WCSS / SSE（聚类紧凑度）
wcss = kmeans.inertia_
print(f"📉 WCSS (Within-Cluster Sum of Squares): {wcss:.2f}")

ch_score = calinski_harabasz_score(X_pca, agg_df['cluster'])
print(f"📈 Calinski-Harabasz Index: {ch_score:.2f}")

explained_var = pca.explained_variance_ratio_

print("📊 PCA Explained Variance Ratio:")
print(f"PCA 1: {explained_var[0]:.4f}")
print(f"PCA 2: {explained_var[1]:.4f}")
print(f"Total Explained Variance (2D): {explained_var.sum():.4f}")

#df.to_csv("output/data_with_combine_cluster.csv", index=False


