"""
hospitality_analysis_py_lab.py
Run in Python Lab / any Python environment.

Requirements:
pip install pandas numpy matplotlib seaborn scikit-learn mlxtend
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# -----------------------
# Config
# -----------------------
INPUT_CSV = "synthetic_hospitality_dataset.csv"   # change path if needed
OUTPUT_DIR = "outputs_py_lab"
os.makedirs(OUTPUT_DIR, exist_ok=True)
PLOT_DPI = 120

# -----------------------
# Helper: save figure
# -----------------------
def save_fig(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, bbox_inches='tight', dpi=PLOT_DPI)
    print(f"Saved: {path}")

# -----------------------
# Load dataset
# -----------------------
if not os.path.exists(INPUT_CSV):
    print(f"ERROR: '{INPUT_CSV}' not found in working directory.")
    print("Place the CSV in the same folder or update INPUT_CSV path in the script.")
    sys.exit(1)

df = pd.read_csv(INPUT_CSV)
print("Dataset loaded. Shape:", df.shape)

# -----------------------
# Quick overview
# -----------------------
print("\n=== HEAD ===")
print(df.head().to_string(index=False))

print("\n=== INFO ===")
print(df.info())

print("\n=== DESCRIBE ===")
print(df.describe(include='all').transpose().head(10).to_string())

# -----------------------
# Preprocessing: missing values
# -----------------------
missing = df.isnull().sum()
print("\nMissing values per column:\n", missing)

# Simple strategy: drop rows with missing values (safe for small synthetic dataset)
if missing.sum() > 0:
    print("\nDropping rows with missing values...")
    df = df.dropna().reset_index(drop=True)
    print("After drop, shape:", df.shape)

# Convert boolean-like extras to 0/1 if they are not
extras = ['Extra_Spa','Extra_Airport_Pickup','Extra_City_Tour','Extra_Sea_View','Extra_Late_Checkout']
for col in extras:
    if col in df.columns:
        # Try to normalize values to 0/1
        df[col] = df[col].replace({True:1, False:0, 'Yes':1, 'No':0, 'yes':1, 'no':0}).fillna(0).astype(int)
    else:
        print(f"Warning: expected extra service column '{col}' not found.")

# -----------------------
# Simple Visualizations (saved)
# -----------------------
sns.set(style="whitegrid")

# Room Type countplot
if 'Room_Type' in df.columns:
    fig = plt.figure(figsize=(7,4))
    ax = sns.countplot(x='Room_Type', data=df, order=df['Room_Type'].value_counts().index)
    ax.set_title("Room Type Preferences")
    plt.xticks(rotation=45)
    save_fig(fig, "room_type_count.png")
    plt.close(fig)

# Booking Channel countplot
if 'Booking_Channel' in df.columns:
    fig = plt.figure(figsize=(7,4))
    ax = sns.countplot(x='Booking_Channel', data=df, order=df['Booking_Channel'].value_counts().index)
    ax.set_title("Booking Channel Distribution")
    plt.xticks(rotation=45)
    save_fig(fig, "booking_channel_count.png")
    plt.close(fig)

# Extras bar chart
present_extras = [c for c in extras if c in df.columns]
if present_extras:
    fig = plt.figure(figsize=(7,4))
    counts = df[present_extras].sum().sort_values(ascending=False)
    counts.plot(kind='bar')
    plt.title("Extra Services Usage (total counts)")
    plt.ylabel("Count")
    save_fig(fig, "extras_usage.png")
    plt.close(fig)

# -----------------------
# Association Rule Mining
# -----------------------
print("\n--- Association Rule Mining ---")
# For association mining, we'll build transactions from extras + categorical columns (Room_Type, Meal_Plan, Booking_Channel)
trans_cols = []

# include extras (already 0/1)
trans_cols.extend([c for c in present_extras])

# For categorical columns - convert to one-hot encoded transaction items
cat_cols = []
for c in ['Room_Type','Meal_Plan','Booking_Channel']:
    if c in df.columns:
        cat_cols.append(c)
        dummies = pd.get_dummies(df[c], prefix=c)
        # join to df_trans
        df = pd.concat([df, dummies], axis=1)
        trans_cols.extend(list(dummies.columns))

if not trans_cols:
    print("No transaction columns available for association mining. Skipping.")
else:
    df_trans = df[trans_cols].astype(int)
    print("Transaction matrix shape:", df_trans.shape)

    # Minimum support: set to 0.05 (5%) or adjust for synthetic size
    min_support = 0.05
    frequent_itemsets = apriori(df_trans, min_support=min_support, use_colnames=True)
    frequent_itemsets = frequent_itemsets.sort_values(by='support', ascending=False).reset_index(drop=True)
    print(f"Found {len(frequent_itemsets)} frequent itemsets with support >= {min_support}")

    # Save top frequent itemsets
    frequent_itemsets.head(20).to_csv(os.path.join(OUTPUT_DIR, "frequent_itemsets_top20.csv"), index=False)
    print("Saved top frequent itemsets to outputs_py_lab/frequent_itemsets_top20.csv")

    # Generate association rules
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
    rules = rules.sort_values(by=['lift','confidence'], ascending=False).reset_index(drop=True)
    print("Generated association rules. Total rules:", len(rules))

    # Save top rules
    rules.to_csv(os.path.join(OUTPUT_DIR, "association_rules_all.csv"), index=False)
    print("Saved all rules to outputs_py_lab/association_rules_all.csv")

    # Print top 5 rules (by lift)
    print("\nTop 5 association rules (by lift):")
    if len(rules) > 0:
        print(rules[['antecedents','consequents','support','confidence','lift']].head(5).to_string(index=False))
    else:
        print("No rules generated. Try lowering min_support or min_threshold.")

# -----------------------
# Customer Segmentation (KMeans)
# -----------------------
print("\n--- Customer Segmentation ---")
num_features = ['Total_Nights','Total_Cost','Num_Adults','Num_Children']
available_num = [c for c in num_features if c in df.columns]

if len(available_num) < 2:
    print("Not enough numerical columns for clustering. Required at least 2 of", num_features)
else:
    features_df = df[available_num].copy()
    # handle any missing or infinite
    features_df = features_df.replace([np.inf, -np.inf], np.nan).dropna()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features_df)

    # Elbow method (1..8 clusters)
    wcss = []
    cluster_range = list(range(1,9))
    for k in cluster_range:
        kmeans_tmp = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans_tmp.fit(X_scaled)
        wcss.append(kmeans_tmp.inertia_)

    # save elbow plot
    fig = plt.figure(figsize=(7,4))
    plt.plot(cluster_range, wcss, marker='o')
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("WCSS (Inertia)")
    plt.title("Elbow Method for K")
    save_fig(fig, "elbow_method.png")
    plt.close(fig)

    # Choose k automatically by simple heuristic: elbow at 3 (common for small dataset).
    # If you want manual change, adjust K_CHOICE below.
    K_CHOICE = 3
    print(f"Using K = {K_CHOICE} for final clustering (change K_CHOICE in script if needed).")

    kmeans = KMeans(n_clusters=K_CHOICE, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)

    # attach cluster labels
    df['Cluster'] = cluster_labels

    # Save cluster csv
    out_csv = os.path.join(OUTPUT_DIR, "customer_segments.csv")
    df.to_csv(out_csv, index=False)
    print("Saved full dataset with cluster labels to:", out_csv)

    # Visualize clusters: pick first two numerical features for 2D scatter
    feat_x = available_num[0]
    feat_y = available_num[1] if len(available_num) > 1 else available_num[0]

    fig = plt.figure(figsize=(7,5))
    ax = sns.scatterplot(x=df[feat_x], y=df[feat_y], hue=df['Cluster'], palette='tab10', s=60)
    ax.set_title(f"Clusters: {feat_x} vs {feat_y}")
    plt.legend(title='Cluster')
    save_fig(fig, f"clusters_{feat_x}_vs_{feat_y}.png")
    plt.close(fig)

    # Boxplot for Total_Cost per cluster (if present)
    if 'Total_Cost' in df.columns:
        fig = plt.figure(figsize=(7,5))
        sns.boxplot(x='Cluster', y='Total_Cost', data=df)
        plt.title("Total Cost distribution per Cluster")
        save_fig(fig, "boxplot_total_cost_per_cluster.png")
        plt.close(fig)

    # Cluster summary (means)
    cluster_summary = df.groupby('Cluster')[available_num].mean().round(2)
    print("\nCluster summary (means):")
    print(cluster_summary.to_string())

    # Save cluster summary
    cluster_summary.to_csv(os.path.join(OUTPUT_DIR, "cluster_summary_means.csv"))
    print("Saved cluster summary to outputs_py_lab/cluster_summary_means.csv")

print("\n--- Done ---")
print(f"All outputs (plots, csvs) saved in the folder: {OUTPUT_DIR}")
