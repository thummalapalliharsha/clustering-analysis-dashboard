import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# 1. Load data
df = pd.read_csv('bank_customers.csv')

# Handle missing values if any
df['BALANCE'] = df['BALANCE'].fillna(df['BALANCE'].median())
df['PURCHASES'] = df['PURCHASES'].fillna(df['PURCHASES'].median())
df['CASH_ADVANCE'] = df['CASH_ADVANCE'].fillna(df['CASH_ADVANCE'].median())

# ONLY the 3 key easy-to-read behavioral features
features_to_cluster = ['BALANCE', 'PURCHASES', 'CASH_ADVANCE']
X = df[features_to_cluster].copy()

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA for 2D visualization (optional since we have 3D native space)
pca_2d = PCA(n_components=2, random_state=42)
X_pca_2d = pca_2d.fit_transform(X_scaled)
df['PCA1'] = X_pca_2d[:, 0]
df['PCA2'] = X_pca_2d[:, 1]

# -------------------------------------------------------------
# 1. K-Means Evaluation
# -------------------------------------------------------------
kmeans_results = {}
for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    ch = calinski_harabasz_score(X_scaled, labels)
    db = davies_bouldin_score(X_scaled, labels)
    kmeans_results[k] = {
        'inertia': float(km.inertia_),
        'silhouette': float(sil),
        'calinski_harabasz': float(ch),
        'davies_bouldin': float(db)
    }

# Train standard K=4 model
best_km = KMeans(n_clusters=4, random_state=42, n_init=10)
df['KMeans_Cluster'] = best_km.fit_predict(X_scaled)

# -------------------------------------------------------------
# 2. Hierarchical (Agglomerative) Evaluation
# -------------------------------------------------------------
hierarchical_results = {}
for k in range(2, 11):
    agg = AgglomerativeClustering(n_clusters=k, linkage='ward')
    labels = agg.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    ch = calinski_harabasz_score(X_scaled, labels)
    db = davies_bouldin_score(X_scaled, labels)
    hierarchical_results[k] = {
        'silhouette': float(sil),
        'calinski_harabasz': float(ch),
        'davies_bouldin': float(db)
    }

best_agg = AgglomerativeClustering(n_clusters=4, linkage='ward')
df['Hierarchical_Cluster'] = best_agg.fit_predict(X_scaled)

# -------------------------------------------------------------
# 3. DBSCAN Evaluation
# -------------------------------------------------------------
dbscan = DBSCAN(eps=0.5, min_samples=15)
df['DBSCAN_Cluster'] = dbscan.fit_predict(X_scaled)

n_db_clusters = len(set(df['DBSCAN_Cluster'])) - (1 if -1 in df['DBSCAN_Cluster'] else 0)
n_db_noise = (df['DBSCAN_Cluster'] == -1).sum()

mask = df['DBSCAN_Cluster'] != -1
unique_core_clusters = set(df['DBSCAN_Cluster'][mask])
if len(unique_core_clusters) >= 2:
    db_sil = float(silhouette_score(X_scaled[mask], df['DBSCAN_Cluster'][mask]))
else:
    db_sil = 0.0

dbscan_summary = {
    'eps': 0.5,
    'min_samples': 15,
    'n_clusters': int(n_db_clusters),
    'n_noise': int(n_db_noise),
    'noise_ratio': float(n_db_noise / len(df)),
    'silhouette_core': float(db_sil)
}

# -------------------------------------------------------------
# Save Outputs & Pre-computed Models
# -------------------------------------------------------------
os.makedirs('clustering_outputs', exist_ok=True)
df.to_csv('clustering_outputs/clustered_customers.csv', index=False)
joblib.dump(scaler, 'clustering_outputs/scaler_3features.joblib')
joblib.dump(best_km, 'clustering_outputs/kmeans_3features_k4.joblib')

with open('clustering_outputs/metrics_3features.json', 'w') as f:
    json.dump({
        'features': features_to_cluster,
        'kmeans': kmeans_results,
        'hierarchical': hierarchical_results,
        'dbscan': dbscan_summary,
        'pca_variance_2d': pca_2d.explained_variance_ratio_.tolist()
    }, f, indent=2)

print("3-Feature Clustering training completed successfully!")
print("Features used:", features_to_cluster)
print("K-Means K=4 Silhouette:", kmeans_results[4]['silhouette'])
print("Hierarchical K=4 Silhouette:", hierarchical_results[4]['silhouette'])
print("DBSCAN clusters:", n_db_clusters, "Noise points:", n_db_noise)
