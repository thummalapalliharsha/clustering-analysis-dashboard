import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# -------------------------------------------------------------
# Streamlit Page Configuration
# -------------------------------------------------------------
st.set_page_config(
    page_title="Bank Customer Segmentation (3 Behaviors)",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #F3F4F6 0%, #FFFFFF 100%);
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
    }
    .persona-card {
        background-color: #F8FAFC;
        border-left: 5px solid #2563EB;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 12px;
    }
    .feature-badge {
        display: inline-block;
        background-color: #EFF6FF;
        color: #1D4ED8;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        margin-right: 6px;
        border: 1px solid #BFDBFE;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 18px;
        border-radius: 8px 8px 0 0;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# Load & Cache Data (3 Features Only)
# -------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv('bank_customers.csv')
    
    # 3 core behaviors
    features = ['BALANCE', 'PURCHASES', 'CASH_ADVANCE']
    
    # Impute missing values if any
    for col in features:
        df[col] = df[col].fillna(df[col].median())
        
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])
    
    pca_2d = PCA(n_components=2, random_state=42)
    X_pca_2d = pca_2d.fit_transform(X_scaled)
    df['PCA1'] = X_pca_2d[:, 0]
    df['PCA2'] = X_pca_2d[:, 1]
    
    return df, features, scaler, X_scaled, pca_2d

df, features, scaler, X_scaled, pca_2d = load_data()

# -------------------------------------------------------------
# Sidebar Configuration
# -------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/bank-cards.png", width=70)
    st.title("Behavioral Engine")
    st.markdown("""
    **Active Features (3 Core Behaviors):**
    - 💰 **`BALANCE`**
    - 🛒 **`PURCHASES`**
    - 🏧 **`CASH_ADVANCE`**
    """)
    st.markdown("---")
    
    st.subheader("Visualization Settings")
    plot_type = st.radio(
        "Plot Dimension:",
        ["3D Native Space (Balance, Purchases, Cash Advance)", "2D Pairwise Projections", "2D PCA Projection"],
        index=0
    )
    sample_size = st.slider("Visualization Sample Count:", min_value=1000, max_value=len(df), value=4500, step=500)
    st.markdown("---")
    st.caption("3-Behavior Simplified Customer Segmentation")

# -------------------------------------------------------------
# Main Application Header
# -------------------------------------------------------------
st.markdown('<div class="main-header">💳 Bank Customer Segmentation Hub</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Clustering over 3 Easy-to-Read Behaviors: '
    '<span class="feature-badge">💰 BALANCE</span>'
    '<span class="feature-badge">🛒 PURCHASES</span>'
    '<span class="feature-badge">🏧 CASH_ADVANCE</span></div>',
    unsafe_allow_html=True
)

# Top KPI Summary Row
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Total Customers", f"{len(df):,}")
with c2:
    st.metric("Avg Balance", f"${df['BALANCE'].mean():,.2f}")
with c3:
    st.metric("Avg Purchases", f"${df['PURCHASES'].mean():,.2f}")
with c4:
    st.metric("Avg Cash Advance", f"${df['CASH_ADVANCE'].mean():,.2f}")

# Subsample Data for Fast Interactive Plotly Rendering
df_plot = df.sample(n=sample_size, random_state=42).copy()

# Helper plotting function
def render_cluster_plot(df_sub, label_col, title_prefix, color_map=None):
    if plot_type == "3D Native Space (Balance, Purchases, Cash Advance)":
        fig = px.scatter_3d(
            df_sub, x='BALANCE', y='PURCHASES', z='CASH_ADVANCE', color=label_col,
            hover_data=['CUST_ID', 'BALANCE', 'PURCHASES', 'CASH_ADVANCE'],
            color_discrete_sequence=px.colors.qualitative.Bold if not color_map else None,
            color_discrete_map=color_map,
            title=f"{title_prefix} - 3D Behavioral Space (Balance vs Purchases vs Cash Advance)"
        )
        fig.update_layout(height=520, margin=dict(l=0, r=0, b=0, t=30))
        st.plotly_chart(fig, use_container_width=True)
        
    elif plot_type == "2D Pairwise Projections":
        p_axis = st.selectbox(
            "Select 2D Axes:",
            ["Balance vs Purchases", "Balance vs Cash Advance", "Purchases vs Cash Advance"],
            key=f"axis_{title_prefix}"
        )
        if p_axis == "Balance vs Purchases":
            x_col, y_col = 'BALANCE', 'PURCHASES'
        elif p_axis == "Balance vs Cash Advance":
            x_col, y_col = 'BALANCE', 'CASH_ADVANCE'
        else:
            x_col, y_col = 'PURCHASES', 'CASH_ADVANCE'
            
        fig = px.scatter(
            df_sub, x=x_col, y=y_col, color=label_col,
            hover_data=['CUST_ID', 'BALANCE', 'PURCHASES', 'CASH_ADVANCE'],
            color_discrete_sequence=px.colors.qualitative.Bold if not color_map else None,
            color_discrete_map=color_map,
            title=f"{title_prefix} - {x_col} vs {y_col}"
        )
        fig.update_layout(height=480)
        st.plotly_chart(fig, use_container_width=True)
        
    else: # 2D PCA
        fig = px.scatter(
            df_sub, x='PCA1', y='PCA2', color=label_col,
            hover_data=['CUST_ID', 'BALANCE', 'PURCHASES', 'CASH_ADVANCE'],
            color_discrete_sequence=px.colors.qualitative.Bold if not color_map else None,
            color_discrete_map=color_map,
            title=f"{title_prefix} - 2D PCA Projection"
        )
        fig.update_layout(height=480)
        st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------
# Navigation Tabs
# -------------------------------------------------------------
tabs = st.tabs([
    "🎯 1. K-Means Clustering",
    "🌳 2. Hierarchical Clustering",
    "🔍 3. DBSCAN (Density-Based)",
    "⚖️ 4. Model Comparison & Personas",
    "📊 5. 3-Behavior EDA Explorer",
    "🔮 6. 3-Behavior Persona Predictor"
])

# =============================================================
# TAB 1: K-MEANS CLUSTERING
# =============================================================
with tabs[0]:
    st.subheader("🎯 K-Means Clustering on 3 Behaviors")
    st.markdown("K-Means groups customers by minimizing Euclidean distances across `BALANCE`, `PURCHASES`, and `CASH_ADVANCE`.")
    
    col_k_ctrl, col_k_metrics = st.columns([1, 2])
    with col_k_ctrl:
        k_val = st.slider("Select Number of Clusters (K):", min_value=2, max_value=8, value=4, step=1, key="km_k_slider")
        km_init = st.selectbox("Initialization:", ["k-means++", "random"], key="km_init_select")
        
        # Fit K-Means
        km = KMeans(n_clusters=k_val, init=km_init, random_state=42, n_init=10)
        km_labels = km.fit_predict(X_scaled)
        df['KMeans_Cluster'] = [f"Cluster {i}" for i in km_labels]
        df_plot['KMeans_Cluster'] = [f"Cluster {i}" for i in km_labels[df_plot.index]]
        
        sil = silhouette_score(X_scaled, km_labels)
        ch = calinski_harabasz_score(X_scaled, km_labels)
        db = davies_bouldin_score(X_scaled, km_labels)
        
    with col_k_metrics:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Silhouette Score", f"{sil:.3f}", help="Score > 0.5 indicates strong, well-separated clusters.")
        m2.metric("Calinski-Harabasz", f"{ch:,.0f}")
        m3.metric("Davies-Bouldin", f"{db:.3f}")
        m4.metric("Inertia (WCSS)", f"{km.inertia_:,.0f}")
        
    st.markdown("---")
    
    col_plot, col_profile = st.columns([3, 2])
    with col_plot:
        render_cluster_plot(df_plot, 'KMeans_Cluster', f"K-Means (K={k_val})")
        
    with col_profile:
        st.markdown("#### Cluster Population Share")
        cluster_counts = df['KMeans_Cluster'].value_counts().reset_index()
        cluster_counts.columns = ['Cluster', 'Count']
        fig_pie = px.pie(
            cluster_counts, values='Count', names='Cluster',
            color_discrete_sequence=px.colors.qualitative.Bold,
            hole=0.4
        )
        fig_pie.update_layout(height=480, margin=dict(t=30, b=20, l=20, r=20))
        st.plotly_chart(fig_pie, use_container_width=True)
        
    st.markdown("#### 📋 3-Behavior Cluster Profiles (Mean Values)")
    km_summary = df.groupby('KMeans_Cluster')[features].agg(['mean', 'median', 'count']).round(2)
    st.dataframe(km_summary.style.background_gradient(cmap='Blues'), use_container_width=True)

# =============================================================
# TAB 2: HIERARCHICAL CLUSTERING
# =============================================================
with tabs[1]:
    st.subheader("🌳 Hierarchical (Agglomerative) Clustering")
    st.markdown("Agglomerative clustering creates bottom-up hierarchical trees over the 3 standardized customer behaviors.")
    
    col_h_ctrl, col_h_metrics = st.columns([1, 2])
    with col_h_ctrl:
        h_k_val = st.slider("Select Number of Clusters (K):", min_value=2, max_value=8, value=4, step=1, key="h_k_slider")
        linkage_method = st.selectbox("Linkage Method:", ["ward", "complete", "average"], key="h_link_select")
        
        # Fit Agglomerative
        agg = AgglomerativeClustering(n_clusters=h_k_val, linkage=linkage_method)
        h_labels = agg.fit_predict(X_scaled)
        df['Hierarchical_Cluster'] = [f"Cluster {i}" for i in h_labels]
        df_plot['Hierarchical_Cluster'] = [f"Cluster {i}" for i in h_labels[df_plot.index]]
        
        h_sil = silhouette_score(X_scaled, h_labels)
        h_ch = calinski_harabasz_score(X_scaled, h_labels)
        h_db = davies_bouldin_score(X_scaled, h_labels)
        
    with col_h_metrics:
        hm1, hm2, hm3 = st.columns(3)
        hm1.metric("Silhouette Score", f"{h_sil:.3f}")
        hm2.metric("Calinski-Harabasz", f"{h_ch:,.0f}")
        hm3.metric("Davies-Bouldin", f"{h_db:.3f}")
        
    st.markdown("---")
    
    col_h_plot, col_h_bar = st.columns([3, 2])
    with col_h_plot:
        render_cluster_plot(df_plot, 'Hierarchical_Cluster', f"Hierarchical ({linkage_method.capitalize()}, K={h_k_val})")
        
    with col_h_bar:
        st.markdown("#### Average Behavioral Metrics by Cluster")
        h_summary = df.groupby('Hierarchical_Cluster')[features].mean().reset_index()
        fig_hbar = px.bar(
            h_summary, x='Hierarchical_Cluster', y=features,
            barmode='group',
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_hbar.update_layout(height=480, legend_title_text='Feature', xaxis_title='Cluster')
        st.plotly_chart(fig_hbar, use_container_width=True)

# =============================================================
# TAB 3: DBSCAN CLUSTERING
# =============================================================
with tabs[2]:
    st.subheader("🔍 DBSCAN (Density-Based Anomaly & Cluster Detection)")
    st.markdown("DBSCAN groups customers residing in dense 3D spatial regions (`BALANCE`, `PURCHASES`, `CASH_ADVANCE`) and isolates outliers as **Noise (-1)**.")
    
    col_db_ctrl, col_db_metrics = st.columns([1, 2])
    with col_db_ctrl:
        eps_val = st.slider("Neighborhood Radius (Epsilon - ε):", min_value=0.2, max_value=2.0, value=0.5, step=0.1)
        min_samp = st.slider("Minimum Samples (min_samples):", min_value=5, max_value=30, value=15, step=1)
        
        # Fit DBSCAN
        dbscan_model = DBSCAN(eps=eps_val, min_samples=min_samp)
        db_labels = dbscan_model.fit_predict(X_scaled)
        
        db_str_labels = [f"Cluster {i}" if i != -1 else "Noise / Anomaly (-1)" for i in db_labels]
        df['DBSCAN_Cluster'] = db_str_labels
        df_plot['DBSCAN_Cluster'] = [db_str_labels[idx] for idx in df_plot.index]
        
        n_clusters_found = len(set(db_labels)) - (1 if -1 in db_labels else 0)
        n_noise_found = (db_labels == -1).sum()
        
    with col_db_metrics:
        dbm1, dbm2, dbm3, dbm4 = st.columns(4)
        dbm1.metric("Clusters Found", f"{n_clusters_found}")
        dbm2.metric("Noise Points", f"{n_noise_found:,}")
        dbm3.metric("Noise Ratio", f"{(n_noise_found/len(df))*100:.1f}%")
        if n_clusters_found > 1:
            core_mask = db_labels != -1
            core_sil = silhouette_score(X_scaled[core_mask], db_labels[core_mask]) if core_mask.sum() > 0 else 0.0
            core_ch = calinski_harabasz_score(X_scaled[core_mask], db_labels[core_mask]) if core_mask.sum() > 0 else 0.0
            core_db = davies_bouldin_score(X_scaled[core_mask], db_labels[core_mask]) if core_mask.sum() > 0 else 0.0
            dbm4.metric("Core Silhouette", f"{core_sil:.3f}")
        else:
            core_sil = None
            core_ch = None
            core_db = None
            dbm4.metric("Core Silhouette", "N/A (1 cluster)")
            
    st.markdown("---")
    
    col_db_plot, col_db_noise = st.columns([3, 2])
    with col_db_plot:
        render_cluster_plot(
            df_plot, 'DBSCAN_Cluster', f"DBSCAN (ε={eps_val}, min={min_samp})",
            color_map={"Noise / Anomaly (-1)": "#EF4444"}
        )
        
    with col_db_noise:
        st.markdown("#### 🚨 Outlier vs Core Behavioral Comparison")
        noise_df = df[df['DBSCAN_Cluster'] == "Noise / Anomaly (-1)"]
        core_df = df[df['DBSCAN_Cluster'] != "Noise / Anomaly (-1)"]
        
        comp_db = pd.DataFrame({
            'Behavior': ['Avg Balance', 'Avg Purchases', 'Avg Cash Advance', 'Customer Count'],
            'Noise / Outliers (-1)': [
                f"${noise_df['BALANCE'].mean():,.2f}",
                f"${noise_df['PURCHASES'].mean():,.2f}",
                f"${noise_df['CASH_ADVANCE'].mean():,.2f}",
                f"{len(noise_df):,}"
            ],
            'Core Clusters': [
                f"${core_df['BALANCE'].mean():,.2f}",
                f"${core_df['PURCHASES'].mean():,.2f}",
                f"${core_df['CASH_ADVANCE'].mean():,.2f}",
                f"{len(core_df):,}"
            ]
        })
        st.dataframe(comp_db, use_container_width=True, hide_index=True)

# =============================================================
# TAB 4: MODEL COMPARISON & PERSONAS
# =============================================================
with tabs[3]:
    st.subheader("⚖️ Model Comparison on 3 Behaviors (Balance, Purchases, Cash Advance)")
    
    # Format metrics for comparison table
    db_sil_str = f"{core_sil:.3f}" if core_sil is not None else "N/A (1 cluster)"
    db_ch_str = f"{core_ch:,.0f}" if core_ch is not None else "N/A (1 cluster)"
    db_db_str = f"{core_db:.3f}" if core_db is not None else "N/A (1 cluster)"
    
    # Calculate best performers
    algo_names = [f"K-Means (K={k_val})", f"Hierarchical ({linkage_method.capitalize()}, K={h_k_val})"]
    sil_vals = [("K-Means", sil), ("Hierarchical", h_sil)]
    ch_vals = [("K-Means", ch), ("Hierarchical", h_ch)]
    db_vals = [("K-Means", db), ("Hierarchical", h_db)]
    
    if core_sil is not None and n_clusters_found > 1:
        algo_names.append(f"DBSCAN (ε={eps_val}, min={min_samp})")
        sil_vals.append(("DBSCAN (Core)", core_sil))
        ch_vals.append(("DBSCAN (Core)", core_ch))
        db_vals.append(("DBSCAN (Core)", core_db))
    else:
        algo_names.append(f"DBSCAN (ε={eps_val}, min={min_samp})")
        
    best_sil_algo, best_sil_score = max(sil_vals, key=lambda x: x[1])
    best_ch_algo, best_ch_score = max(ch_vals, key=lambda x: x[1])
    best_db_algo, best_db_score = min(db_vals, key=lambda x: x[1])
    
    comp_data = {
        "Clustering Algorithm": [
            f"K-Means (K={k_val})",
            f"Hierarchical ({linkage_method.capitalize()}, K={h_k_val})",
            f"DBSCAN (ε={eps_val}, min={min_samp})"
        ],
        "Clusters (K)": [k_val, h_k_val, f"{n_clusters_found} core"],
        "Silhouette Score (↑ Higher)": [f"{sil:.3f}", f"{h_sil:.3f}", db_sil_str],
        "Davies-Bouldin Score (↓ Lower)": [f"{db:.3f}", f"{h_db:.3f}", db_db_str],
        "Calinski-Harabasz Score (↑ Higher)": [f"{ch:,.0f}", f"{h_ch:,.0f}", db_ch_str],
        "Outliers Handled": ["None (All assigned)", "None (All assigned)", f"{n_noise_found:,} ({n_noise_found/len(df)*100:.1f}%)"]
    }
    
    st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)
    
    # Best Performer Callout Cards
    st.markdown("#### 🏆 Performance Benchmark & Best Model Selection")
    col_win1, col_win2, col_win3 = st.columns(3)
    with col_win1:
        st.markdown(f"""
        <div class="metric-card" style="border-left: 5px solid #10B981;">
            <div style="font-size: 0.85rem; color: #6B7280; font-weight: 600;">BEST SILHOUETTE SCORE (↑)</div>
            <div style="font-size: 1.3rem; font-weight: 800; color: #10B981; margin: 4px 0;">{best_sil_algo}</div>
            <div style="font-size: 0.95rem; font-weight: 700;">Score: {best_sil_score:.3f}</div>
            <div style="font-size: 0.75rem; color: #4B5563;">Measures cluster cohesion & separation (-1 to 1)</div>
        </div>
        """, unsafe_allow_html=True)
    with col_win2:
        st.markdown(f"""
        <div class="metric-card" style="border-left: 5px solid #3B82F6;">
            <div style="font-size: 0.85rem; color: #6B7280; font-weight: 600;">BEST DAVIES-BOULDIN (↓)</div>
            <div style="font-size: 1.3rem; font-weight: 800; color: #3B82F6; margin: 4px 0;">{best_db_algo}</div>
            <div style="font-size: 0.95rem; font-weight: 700;">Score: {best_db_score:.3f}</div>
            <div style="font-size: 0.75rem; color: #4B5563;">Measures similarity between clusters (Lower is better)</div>
        </div>
        """, unsafe_allow_html=True)
    with col_win3:
        st.markdown(f"""
        <div class="metric-card" style="border-left: 5px solid #8B5CF6;">
            <div style="font-size: 0.85rem; color: #6B7280; font-weight: 600;">BEST CALINSKI-HARABASZ (↑)</div>
            <div style="font-size: 1.3rem; font-weight: 800; color: #8B5CF6; margin: 4px 0;">{best_ch_algo}</div>
            <div style="font-size: 0.95rem; font-weight: 700;">Score: {best_ch_score:,.0f}</div>
            <div style="font-size: 0.75rem; color: #4B5563;">Variance ratio between vs within clusters</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 💡 The 4 Core Behavioral Personas (K=4 Standard)")
    
    p1, p2 = st.columns(2)
    with p1:
        st.markdown("""
        <div class="persona-card">
            <h4>🛒 1. Active Purchasers / Transactors</h4>
            <p><b>Behavior:</b> <b>High Purchases</b> ($3,500+), <b>Low/Moderate Balance</b> (<$1,000), <b>Zero/Low Cash Advance</b> ($0 - $100).</p>
            <p><b>Business Action:</b> Premium merchant cash-back rewards, travel perks, and credit line increases to maximize interchange fee revenue.</p>
        </div>
        <div class="persona-card">
            <h4>🏧 2. Cash Advance Borrowers</h4>
            <p><b>Behavior:</b> <b>High Cash Advance</b> ($4,000+), <b>High Balance</b> ($4,500+), <b>Low Retail Purchases</b> (<$500).</p>
            <p><b>Business Action:</b> High risk of delinquency. Offer fixed-rate personal debt consolidation loans and lower cash advance withdrawal caps.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with p2:
        st.markdown("""
        <div class="persona-card">
            <h4>🔄 3. Debt Revolvers / High Balance</h4>
            <p><b>Behavior:</b> <b>High Balance</b> ($3,500+), <b>Low Purchases</b> (<$500), <b>Low Cash Advance</b> (<$500).</p>
            <p><b>Business Action:</b> Primary interest margin revenue generators. Offer balance transfer promotions with attractive 0% APR windows.</p>
        </div>
        <div class="persona-card">
            <h4>💤 4. Low-Activity / Budget Customers</h4>
            <p><b>Behavior:</b> <b>Low Balance</b> (<$500), <b>Low Purchases</b> (<$300), <b>Zero Cash Advance</b> ($0).</p>
            <p><b>Business Action:</b> Re-engagement spending challenges (e.g. spend $50 this month for $10 bonus) to become top-of-wallet.</p>
        </div>
        """, unsafe_allow_html=True)

# =============================================================
# TAB 5: 3-BEHAVIOR EDA EXPLORER
# =============================================================
with tabs[4]:
    st.subheader("📊 3-Behavior Exploratory Data Analysis")
    
    c_eda1, c_eda2 = st.columns(2)
    with c_eda1:
        st.markdown("#### Correlation Matrix")
        corr = df[features].corr()
        fig_heat = px.imshow(
            corr, text_auto=".2f", color_continuous_scale="Blues",
            zmin=0, zmax=1, aspect="auto"
        )
        fig_heat.update_layout(height=380)
        st.plotly_chart(fig_heat, use_container_width=True)
        
    with c_eda2:
        st.markdown("#### Distribution Histogram")
        selected_feat = st.selectbox("Select Feature:", features, index=0)
        fig_h = px.histogram(
            df, x=selected_feat, nbins=40, marginal="box",
            color_discrete_sequence=['#2563EB'],
            title=f"Distribution of {selected_feat}"
        )
        fig_h.update_layout(height=380)
        st.plotly_chart(fig_h, use_container_width=True)
        
    st.markdown("#### 📄 3-Behavior Summary Statistics")
    st.dataframe(df[features].describe().T[['mean', 'std', 'min', '25%', '50%', '75%', 'max']].round(2), use_container_width=True)

# =============================================================
# TAB 6: 3-BEHAVIOR LIVE PERSONA PREDICTOR
# =============================================================
with tabs[5]:
    st.subheader("🔮 3-Behavior Real-Time Customer Persona Predictor")
    st.markdown("Adjust the 3 behavioral inputs below to classify any customer profile into its behavioral cluster in real time!")
    
    col_in1, col_in2, col_in3 = st.columns(3)
    with col_in1:
        in_balance = st.number_input("💰 Customer Balance ($):", min_value=0.0, max_value=25000.0, value=1200.0, step=100.0)
    with col_in2:
        in_purchases = st.number_input("🛒 Purchases Total ($):", min_value=0.0, max_value=50000.0, value=3200.0, step=100.0)
    with col_in3:
        in_cash_adv = st.number_input("🏧 Cash Advance ($):", min_value=0.0, max_value=50000.0, value=0.0, step=100.0)
        
    if st.button("🚀 Classify Customer Persona", type="primary", use_container_width=True):
        user_raw = np.array([[in_balance, in_purchases, in_cash_adv]])
        user_scaled = scaler.transform(user_raw)
        pred_cluster = km.predict(user_scaled)[0]
        
        st.success(f"### 🎯 Assigned to: **Cluster {pred_cluster}**")
        
        # Determine persona rule heuristic
        if in_purchases > 2500 and in_cash_adv < 1000:
            persona_name = "🛒 Active Purchaser / Transactor"
            persona_desc = "High transaction volume, drives interchange fee revenue. Recommend premium cash-back and travel reward programs."
        elif in_cash_adv > 2500:
            persona_name = "🏧 Cash Advance Borrower"
            persona_desc = "Relies heavily on ATM cash advances. High credit risk. Recommend fixed-APR debt consolidation."
        elif in_balance > 2500 and in_purchases < 1000:
            persona_name = "🔄 Debt Revolver / High Balance"
            persona_desc = "Maintains elevated revolving balance. High interest margin generator. Offer 0% balance transfer promotion."
        else:
            persona_name = "💤 Low-Activity / Budget Customer"
            persona_desc = "Low card usage. Recommend promotional activation incentives."
            
        st.info(f"**Identified Persona:** {persona_name}\n\n**Strategy:** {persona_desc}")
        
        # 3D visualization showing customer point
        fig_user = px.scatter_3d(
            df_plot, x='BALANCE', y='PURCHASES', z='CASH_ADVANCE', color='KMeans_Cluster',
            opacity=0.3,
            title="Customer Position in 3D Behavioral Space"
        )
        fig_user.add_trace(go.Scatter3d(
            x=[in_balance], y=[in_purchases], z=[in_cash_adv],
            mode='markers+text',
            marker=dict(size=12, color='red', symbol='diamond'),
            name='New Customer Target',
            text=['Target Profile'],
            textposition='top center'
        ))
        fig_user.update_layout(height=500, margin=dict(l=0, r=0, b=0, t=30))
        st.plotly_chart(fig_user, use_container_width=True)
