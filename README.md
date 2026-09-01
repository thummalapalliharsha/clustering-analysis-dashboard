# 💳 Bank Customer Clustering & Financial Analytics Dashboard

[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit%201.61-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn%201.9-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Visuals-Plotly%207.0-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com/)
[![Pandas](https://img.shields.io/badge/Data-Pandas%203.0-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)

An end-to-end unsupervised machine learning and financial analytics application that segments **8,950 credit card customers** into actionable behavioral personas. The project trains and compares **3 clustering algorithms** (K-Means, Hierarchical/Agglomerative, and DBSCAN) focused on core financial behaviors (**`BALANCE`**, **`PURCHASES`**, **`CASH_ADVANCE`**), backed by a modern, responsive **Streamlit** dashboard with 3D interactive visualizations and a real-time persona classifier.

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Key Objectives](#-key-objectives)
- [Dataset Overview](#-dataset-overview)
- [Core Financial Behaviors](#-core-financial-behaviors)
- [Clustering Algorithms Implemented](#-clustering-algorithms-implemented)
- [Evaluation Metrics Deep-Dive](#-evaluation-metrics-deep-dive)
- [Customer Personas & Business Strategies](#-customer-personas--business-strategies)
- [Supervised Dataset Preparation (Regression & Classification)](#-supervised-dataset-preparation)
- [Streamlit Dashboard Features](#-streamlit-dashboard-features)
- [Project Directory Structure](#-project-directory-structure)
- [Installation & Setup](#-installation--setup)
- [Running the Streamlit Application](#-running-the-streamlit-application)
- [Automated Verification & Pipeline Scripts](#-automated-verification--pipeline-scripts)
- [Troubleshooting & Gotchas](#-troubleshooting--gotchas)
- [GitHub Repository](#-github-repository)

---

## 🚀 Project Overview

Financial institutions and credit card issuers manage diverse customer bases with vastly different spending habits, risk profiles, and payment frequencies. Understanding customer behavior without pre-existing labels is critical for:
- **Risk Mitigation**: Detecting customers dependent on ATM cash advances who pose high default risks.
- **Revenue Optimization**: Identifying high-volume spenders to incentivize via targeted rewards and credit line increases.
- **Interest Income Strategy**: Engaging balance revolvers with strategic 0% APR balance transfer promotions.

This repository provides the full pipeline: from raw dataset cleaning, domain feature engineering, and exploratory data analysis (EDA), to unsupervised clustering modeling, mathematical validation, and an interactive web frontend.

---

## 🎯 Key Objectives

1. **Perform In-Depth EDA**: Analyze distributions, skewness, outliers, and collinearities across 8,950 customer records.
2. **Train 3 Unsupervised Clustering Methods**: Implement **K-Means**, **Hierarchical (Agglomerative)**, and **DBSCAN** algorithms on standardized behavioral data.
3. **Compare Clustering Performance**: Benchmark algorithms using **Silhouette Score**, **Davies-Bouldin Index**, and **Calinski-Harabasz Score**.
4. **Isolate Financial Anomalies**: Leverage DBSCAN's density partitioning to isolate outlier accounts (Noise label `-1`).
5. **Interactive Web Application**: Build a production-grade **Streamlit** dashboard featuring 3D rotatable scatter plots, 2D pairwise views, dynamic hyperparameter sliders, and real-time customer persona classification.
6. **Prepare Supervised Datasets**: Generate clean, leak-free train/validation/test splits for downstream **Regression** (`CREDIT_LIMIT`) and **Classification** (`IS_FULL_PAYER`).

---

## 📊 Dataset Overview

- **Source File**: `bank_customers.csv`
- **Total Records**: 8,950 customer profiles
- **Total Features**: 18 original columns capturing 6–12 months of transactional credit history

### Data Dictionary

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `CUST_ID` | `string` | Unique customer identification string |
| `BALANCE` | `float` | Outstanding balance amount left in the account to make purchases |
| `BALANCE_FREQUENCY` | `float` | How frequently the Balance is updated (score between 0 and 1) |
| `PURCHASES` | `float` | Total amount of purchases made from account |
| `ONEOFF_PURCHASES` | `float` | Maximum purchase amount done in one-go |
| `INSTALLMENTS_PURCHASES` | `float` | Amount of purchase done in installment |
| `CASH_ADVANCE` | `float` | Cash in advance given by the user (ATM withdrawals) |
| `PURCHASES_FREQUENCY` | `float` | How frequently the Purchases are being made (0 to 1) |
| `ONEOFF_PURCHASES_FREQUENCY`| `float` | How frequently Purchases are happening in one-go (0 to 1) |
| `PURCHASES_INSTALLMENTS_FREQUENCY` | `float` | How frequently purchases in installments are being done (0 to 1) |
| `CASH_ADVANCE_FREQUENCY` | `float` | How frequently the cash in advance is being paid (0 to 1) |
| `CASH_ADVANCE_TRX` | `integer` | Number of Transactions made with "Cash in Advance" |
| `PURCHASES_TRX` | `integer` | Number of purchase transactions made |
| `CREDIT_LIMIT` | `float` | Limit of Credit Card for user (1 missing value handled via median) |
| `PAYMENTS` | `float` | Amount of Payment done by user |
| `MINIMUM_PAYMENTS` | `float` | Minimum amount of payments made by user (313 missing values handled via median) |
| `PRC_FULL_PAYMENT` | `float` | Percent of full payment paid by user (0 to 1) |
| `TENURE` | `integer` | Tenure of credit card service for user (6 to 12 months) |

---

## 💡 Core Financial Behaviors

To ensure high interpretability and strong geometric cluster separation, the clustering engine focuses on **3 primary, easy-to-read financial behaviors**:

```
                              ┌────────────────────────┐
                              │     BANK CUSTOMER      │
                              └───────────┬────────────┘
                                          │
            ┌─────────────────────────────┼─────────────────────────────┐
            ▼                             ▼                             ▼
   💰 BALANCE                    🛒 PURCHASES                  🏧 CASH_ADVANCE
Outstanding Debt Balance      Retail Transaction Volume      ATM Withdrawal Volume
```

### Why 3 Features?
- **Intuitive Geometric Space**: Allows direct, uncompressed 3D physical visualization $(X=\text{Balance}, Y=\text{Purchases}, Z=\text{Cash Advance})$ without loss of variance from dimensionality reduction.
- **Superior Cluster Separation**: Elevates the **Silhouette Score from 0.19 (high-dimensional space) to 0.557**, creating crisp, non-overlapping behavioral boundaries.

---

## 🔬 Clustering Algorithms Implemented

### 1. 🎯 K-Means Clustering
- **Concept**: Partitions $N$ observations into $K$ spherical clusters by iteratively minimizing the Within-Cluster Sum of Squares (Inertia / WCSS):
  $$\text{WCSS} = \sum_{i=1}^{K} \sum_{x \in S_i} ||x - \mu_i||^2$$
- **Implementation**: `sklearn.cluster.KMeans(n_clusters=K, init='k-means++', n_init=10)`
- **Tuning**: Evaluated dynamically across $K \in [2, 8]$ with Elbow & Silhouette curve analysis.

### 2. 🌳 Hierarchical (Agglomerative) Clustering
- **Concept**: Bottom-up hierarchical tree construction merging pairs of clusters based on linkage distance.
- **Implementation**: `sklearn.cluster.AgglomerativeClustering(n_clusters=K, linkage='ward')`
- **Supported Linkages**:
  - `ward`: Minimizes the variance of clusters being merged (default).
  - `complete`: Maximum pairwise distance between observations of two sets.
  - `average`: Average pairwise distance between observations.

### 3. 🔍 DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
- **Concept**: Discovers dense neighborhoods of arbitrary geometric shape and flags low-density isolated points as **Noise / Anomalies (`-1`)**.
- **Implementation**: `sklearn.cluster.DBSCAN(eps=ε, min_samples=min_samp)`
- **Parameters**:
  - $\varepsilon$ (Epsilon): Maximum neighborhood distance radius (interactive range: `0.2` to `2.0`).
  - $\text{min\_samples}$: Minimum core point threshold (interactive range: `5` to `30`).

---

## 📈 Evaluation Metrics Deep-Dive

The dashboard benchmarks all three clustering methods against standard unsupervised metrics:

| Metric | Mathematical Objective | Ideal Range | Optimization Direction | Formula / Concept |
| :--- | :--- | :---: | :---: | :--- |
| **Silhouette Score** | Measures cohesion within own cluster vs separation from neighboring clusters | $[-1, +1]$ | **Higher is Better (↑)** | $s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$ |
| **Davies-Bouldin Index** | Average similarity between each cluster and its most similar one | $[0, \infty)$ | **Lower is Better (↓)** | $DB = \frac{1}{k} \sum_{i=1}^{k} \max_{j \neq i} \left( \frac{\sigma_i + \sigma_j}{d(c_i, c_j)} \right)$ |
| **Calinski-Harabasz Index** | Ratio of between-cluster dispersion to within-cluster dispersion (Variance Ratio) | $[0, \infty)$ | **Higher is Better (↑)** | $CH = \frac{\text{SS}_B / (k - 1)}{\text{SS}_W / (N - k)}$ |

### Performance Benchmark (3-Behavior Space)

| Clustering Algorithm | Clusters ($K$) | Silhouette Score (↑) | Davies-Bouldin (↓) | Calinski-Harabasz (↑) | Outliers Handled |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **K-Means** | 4 | **0.557** 🥇 | **0.781** 🥇 | **8,452** 🥇 | None (All assigned) |
| **Hierarchical (Ward)** | 4 | 0.504 | 0.812 | 7,120 | None (All assigned) |
| **DBSCAN ($\varepsilon=0.5, \text{min}=15$)** | 2 Core | 0.485 (Core) | 0.894 (Core) | Density-Based | **415 Outliers (4.6%)** 🚨 |

---

## 👥 Customer Personas & Business Strategies

Based on the 3-behavior K-Means partitioning ($K=4$), customer accounts resolve into 4 distinct business segments:

```
                                  4 CUSTOMER PERSONAS
   ┌──────────────────────┬──────────────────────┬──────────────────────┬──────────────────────┐
   │  🛒 ACTIVE SPENDER   │  🏧 CASH ADVANCE     │  🔄 DEBT REVOLVER    │  💤 BUDGET USER      │
   ├──────────────────────┼──────────────────────┼──────────────────────┼──────────────────────┤
   │ Purchases:  >$3,500  │ Purchases:  <$500    │ Purchases:  <$500    │ Purchases:  <$300    │
   │ Balance:    <$1,000  │ Balance:    >$4,500  │ Balance:    >$3,500  │ Balance:    <$500    │
   │ Cash Adv:   $0-$100  │ Cash Adv:   >$4,000  │ Cash Adv:   <$500    │ Cash Adv:   $0       │
   └──────────────────────┴──────────────────────┴──────────────────────┴──────────────────────┘
```

### Strategic Action Plan

1. **🛒 Active Purchasers / Transactors**:
   - **Characteristics**: High purchase volume, low debt balance, zero cash advance.
   - **Business Action**: Premium merchant cash-back rewards, airline mileage perks, and credit limit expansion to maximize interchange fee revenue.
2. **🏧 Cash Advance Borrowers**:
   - **Characteristics**: High ATM cash withdrawals, elevated balances, low retail spending.
   - **Business Action**: High delinquency risk. Lower cash advance withdrawal limits, increase monitoring, and offer fixed-APR debt consolidation personal loans.
3. **🔄 Debt Revolvers / High Balance**:
   - **Characteristics**: Maintains high revolving balance, pays minimums, low purchases.
   - **Business Action**: Primary interest margin generators. Offer attractive 0% APR balance transfer promotions and installment restructuring plans.
4. **💤 Low-Activity / Budget Customers**:
   - **Characteristics**: Low balance, infrequent purchases, zero cash advance (~55% of customer base).
   - **Business Action**: Targeted activation campaigns (e.g., "Spend $50 this month to get $10 statement credit") to achieve top-of-wallet status.

---

## 📦 Supervised Dataset Preparation

In addition to clustering, the project prepares fully preprocessed, leak-free datasets for downstream supervised predictive modeling in [`processed_data/`](processed_data/):

```
processed_data/
├── full_cleaned_engineered.csv              # Full dataset (8,950 rows, 27 columns, 0 NaNs)
├── descriptive_statistics.csv               # Statistical profiling, skewness, kurtosis
├── regression_train.csv                     # Scaled train set (6,266 rows) - Target: CREDIT_LIMIT
├── regression_val.csv                       # Scaled val set   (1,342 rows)
├── regression_test.csv                      # Scaled test set  (1,342 rows)
├── classification_train.csv                 # Scaled train set (6,266 rows) - Target: IS_FULL_PAYER
├── classification_val.csv                   # Scaled val set   (1,342 rows)
├── classification_test.csv                  # Scaled test set  (1,342 rows)
├── regression_preprocessor_meta.json        # Scaler & encoder parameters for regression
├── classification_preprocessor_meta.json    # Scaler & encoder parameters for classification
└── raw_splits/                              # Unscaled raw splits for Tree models (XGBoost/LightGBM)
    ├── reg_raw_train.csv / val / test
    └── clf_raw_train.csv / val / test
```

### Feature Engineering Details:
- `BALANCE_TO_LIMIT_RATIO`: Credit utilization ratio (`BALANCE / CREDIT_LIMIT`).
- `PAYMENT_TO_MIN_PAYMENT_RATIO`: Payment safety buffer (`PAYMENTS / MINIMUM_PAYMENTS`).
- `MONTHLY_AVG_PURCHASES`: Tenure-normalized monthly spending (`PURCHASES / TENURE`).
- `MONTHLY_AVG_CASH_ADVANCE`: Monthly cash advance frequency (`CASH_ADVANCE / TENURE`).
- `CASH_ADVANCE_SHARE`: Ratio of cash advance to total expenditure.
- `PURCHASE_TYPE`: Categorical typology (`Both`, `Installments_Only`, `One_Off_Only`, `No_Purchases`).
- `IS_FULL_PAYER`: Binary classification target (`1` if `PRC_FULL_PAYMENT >= 0.10`, else `0`).

---

## 🖥️ Streamlit Dashboard Features

The web frontend (`app.py`) provides 6 interactive tabs:

1. **🎯 Tab 1: K-Means Clustering**:
   - Dynamic $K$ slider ($K \in [2, 8]$) and initialization options (`k-means++`, `random`).
   - Live metrics (Silhouette, Calinski-Harabasz, Davies-Bouldin, Inertia).
   - Interactive 3D/2D Plotly scatter plots with hover metadata and cluster share pie chart.
2. **🌳 Tab 2: Hierarchical Clustering**:
   - Linkage selection (`ward`, `complete`, `average`) and cluster count slider.
   - Grouped bar charts comparing average Balance, Purchases, and Cash Advance.
3. **🔍 Tab 3: DBSCAN Anomaly Detection**:
   - Dynamic $\varepsilon$ (Epsilon) and $\text{min\_samples}$ sliders.
   - Outlier diagnostics table comparing anomalous accounts against core customer segments.
4. **⚖️ Tab 4: Model Comparison & Personas**:
   - Head-to-head comparison matrix with automated **Best Model Benchmark cards** (Silhouette, Davies-Bouldin, Calinski-Harabasz).
   - Interactive persona strategy cards with actionable recommendations.
5. **📊 Tab 5: 3-Behavior EDA Explorer**:
   - Interactive correlation matrix, feature distribution histogram with box plot marginals, and summary statistics.
6. **🔮 Tab 6: Real-Time Customer Persona Classifier**:
   - Live numerical input cards (`BALANCE`, `PURCHASES`, `CASH_ADVANCE`).
   - Instant cluster prediction and real-time 3D scatter plot highlighting the customer's position.

---

## 📁 Project Directory Structure

```
clustering-analysis-dashboard/
├── .venv/                                   # Python virtual environment
├── clustering_outputs/                      # Trained models and cached clustering artifacts
│   ├── clustered_customers.csv              # Customer dataset with assigned cluster labels
│   ├── kmeans_3features_k4.joblib           # Trained K-Means model (3 features, K=4)
│   ├── kmeans_model_k4.joblib               # Full-dimensional K-Means model
│   ├── metrics.json                         # Full-dimensional clustering metrics
│   ├── metrics_3features.json               # 3-behavior clustering metrics
│   ├── pca_2d.joblib / pca_3d.joblib        # PCA transformation pipelines
│   └── scaler_3features.joblib              # StandardScaler fitted on 3 behaviors
├── eda_plots/                               # High-resolution EDA figures
│   ├── 01_feature_distributions.png         # Univariate histogram distributions
│   ├── 02_correlation_heatmap.png           # Feature correlation matrix
│   ├── 03_outlier_boxplots.png              # Boxplot outlier diagnostics
│   ├── 04_behavioral_relationships.png      # Scatter relationships (Purchases vs Payments, etc.)
│   └── 05_engineered_feature_insights.png   # Engineered ratios & typology insights
├── processed_data/                          # Cleaned, engineered, and split ML datasets
│   ├── raw_splits/                          # Raw CSV splits for tree-based algorithms
│   ├── classification_train/val/test.csv    # Scaled splits for classification
│   ├── regression_train/val/test.csv        # Scaled splits for regression
│   ├── full_cleaned_engineered.csv          # Master clean dataset
│   └── *.json                               # Preprocessor metadata
├── app.py                                   # Streamlit web application frontend
├── bank_customers.csv                       # Original raw customer dataset
├── eda_and_data_preparation.ipynb           # Comprehensive Jupyter Notebook
├── generate_notebook.py                     # Script to build eda_and_data_preparation.ipynb
├── perform_eda_and_prep.py                  # End-to-end data processing & split pipeline
├── requirements.txt                         # Pinned Python package dependencies
├── train_clusters.py                        # Clustering model training & evaluation engine
├── verify_prepared_data.py                  # Automated test suite validating data integrity
└── README.md                                # Comprehensive documentation
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/thummalapalliharsha/clustering-analysis-dashboard.git
cd clustering-analysis-dashboard
```

### 2. Set Up a Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Streamlit Application

Launch the Streamlit interactive dashboard locally:

```powershell
# From within the active virtual environment:
streamlit run app.py
```

Or specify an explicit port:
```powershell
streamlit run app.py --server.port 8503
```

Once started, open your browser and navigate to:
👉 **`http://localhost:8503`** (or `http://localhost:8501` / `http://localhost:8502`)

---

## 🧪 Automated Verification & Pipeline Scripts

### 1. Run Data Preparation & Feature Engineering
Generates all cleaned datasets, scaled splits, and figures in `eda_plots/`:
```bash
python perform_eda_and_prep.py
```

### 2. Train Clustering Models
Trains K-Means, Hierarchical, and DBSCAN algorithms and exports cached models to `clustering_outputs/`:
```bash
python train_clusters.py
```

### 3. Execute Automated Verification Suite
Validates zero missing values, dataset shape alignment, stratification ratios, and artifact generation:
```bash
python verify_prepared_data.py
```

**Expected Output:**
```
--- Starting Automated Data & Artifact Verification ---
[OK] All 5 EDA visualization plots verified.
[OK] Full cleaned dataset verified (8950 rows, 27 columns, 0 NaNs).
[OK] Regression splits verified (Train: (6266, 25), Val: (1342, 25), Test: (1342, 25)).
[OK] Stratified positive class ratios: Train=0.2758, Val=0.2757, Test=0.2757
[OK] Raw unscaled splits for tree models verified.
[OK] Jupyter Notebook verified.

--- ALL VERIFICATIONS PASSED SUCCESSFULLY! ---
```

---

## 🔧 Troubleshooting & Gotchas

1. **Port Already in Use (`Port 8501 is not available`)**:
   - Run on an alternative port:
     ```powershell
     streamlit run app.py --server.port 8504
     ```
2. **Missing `plotly` in Virtual Environment**:
   - If running inside a new `.venv`, install Plotly:
     ```powershell
     pip install plotly
     ```
3. **DBSCAN Single Cluster Warning**:
   - In DBSCAN, if $\varepsilon$ is set too large (e.g. $> 2.5$), all core points may merge into 1 cluster. The app automatically handles this and indicates `N/A (1 cluster)` for the silhouette score. Use the slider to set $\varepsilon \approx 0.4 - 0.7$ for optimal multi-cluster density discovery.

---

## 🌐 GitHub Repository

- **Repository**: [thummalapalliharsha/clustering-analysis-dashboard](https://github.com/thummalapalliharsha/clustering-analysis-dashboard.git)
- **Branch**: `main`

---

## 📜 License

This project is licensed under the Apache-2.0 / MIT License.
