import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the dataset
try:
    df = pd.read_csv('C_origin.csv')
    print("Dataset loaded successfully.")
except Exception as e:
    print(f"Error loading dataset: {e}")

# 1. Data Cleaning & Preprocessing
# Convert 'placement' to numeric, forcing errors to NaN (in case of weird strings)
# Some datasets use 'Eliminated' or other text in placement, so coerce is safer.
df['placement'] = pd.to_numeric(df['placement'], errors='coerce')

# Drop rows where placement is NaN (we can't analyze placement if it doesn't exist)
df_clean = df.dropna(subset=['placement']).copy()

print(f"Original rows: {len(df)}, Rows after dropping NaN placement: {len(df_clean)}")

# Features to analyze against placement
# Note: 'celebrity_homestate' and 'celebrity_industry' are categorical.
# 'celebrity_age_during_season' is numerical.
features = ['celebrity_homestate', 'celebrity_industry', 'celebrity_age_during_season']

# Initialize results storage
results = {}

# Set up the plotting area
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.set(style="whitegrid")

for i, feature in enumerate(features):
    # Prepare temporary dataframe for this pair
    temp_df = df_clean[['placement', feature]].copy()
    
    # Handle missing values in feature and Encode
    if temp_df[feature].dtype == 'object':
        temp_df[feature] = temp_df[feature].fillna('Unknown')
        le = LabelEncoder()
        # Create a new column for the encoded version
        temp_df[f'{feature}_encoded'] = le.fit_transform(temp_df[feature].astype(str))
        calc_feature = f'{feature}_encoded'
        original_feature = feature
    else:
        # Fill numeric NaNs with median
        temp_df[feature] = temp_df[feature].fillna(temp_df[feature].median())
        calc_feature = feature
        original_feature = feature
        
    # Standardize the data (important for K-Means to treat axes equally)
    scaler = StandardScaler()
    X = temp_df[[calc_feature, 'placement']]
    X_scaled = scaler.fit_transform(X)
    
    # Apply K-Means
    # Using k=3 (e.g., Top tier, Middle tier, Low tier)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    temp_df['cluster'] = kmeans.fit_predict(X_scaled)
    
    # Visualization
    ax = axes[i]
    
    # Scatter plot
    scatter = ax.scatter(temp_df[calc_feature], temp_df['placement'], 
                         c=temp_df['cluster'], cmap='viridis', alpha=0.6, edgecolors='w')
    
    ax.set_title(f'Placement vs {feature}\n(K-Means, k=3)')
    ax.set_ylabel('Placement (1 is Best)')
    # Invert y-axis because placement 1 is better than 10, visualizing "higher is better" typically means "up"
    ax.invert_yaxis() 
    
    # Label handling
    if feature == 'celebrity_age_during_season':
        ax.set_xlabel('Age')
    else:
        ax.set_xlabel(f'{feature} (Encoded)')
        # Hide x-ticks for high cardinality categorical data to avoid clutter
        ax.set_xticks([]) 

plt.tight_layout()
plt.savefig('kmeans_analysis.png')
print("Analysis complete. Plot saved as 'kmeans_analysis.png'.")

# Print a small sample of the processed data to show the user what happened
print("\nSample of processed data (Placement + Age):")
print(df_clean[['placement', 'celebrity_age_during_season']].head())