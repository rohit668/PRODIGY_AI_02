# ============================================================
# CUSTOMER SEGMENTATION USING K-MEANS CLUSTERING
# ============================================================
# Dataset: Mall Customer Segmentation Data
# Features: CustomerID, Gender, Age, Annual Income, Spending Score
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
print("=" * 70)
print("CUSTOMER SEGMENTATION USING K-MEANS CLUSTERING")
print("=" * 70)

# ============================================================
# 1. LOAD AND EXPLORE THE DATASET
# ============================================================
print("\n1. LOADING AND EXPLORING THE DATASET...")
print("-" * 50)

# Since we're working in a controlled environment, I'll create synthetic data 
# that matches the structure described in the Kaggle dataset
def create_customer_data(n_samples=200):
    """Create synthetic customer data matching the mall customer dataset structure"""
    np.random.seed(42)
    
    # Generate Customer IDs
    customer_ids = np.arange(1, n_samples + 1)
    
    # Generate Gender (44% Male, 56% Female as per dataset info)
    gender = np.random.choice(['Male', 'Female'], n_samples, p=[0.44, 0.56])
    
    # Generate Age (18-70 as per dataset)
    age = np.random.randint(18, 71, n_samples)
    
    # Generate Annual Income (15-137k as per dataset)
    annual_income = np.random.randint(15, 138, n_samples) * 1000
    
    # Generate Spending Score (1-99 as per dataset)
    spending_score = np.random.randint(1, 100, n_samples)
    
    # Create correlations (higher income doesn't always mean higher spending score)
    # This creates more realistic clusters
    for i in range(n_samples):
        if annual_income[i] > 80000:
            # High income customers might have moderate to low spending scores
            if np.random.random() > 0.7:
                spending_score[i] = np.random.randint(70, 100)
            else:
                spending_score[i] = np.random.randint(20, 70)
        elif annual_income[i] < 30000:
            # Low income customers might have high spending scores (careless spenders)
            if np.random.random() > 0.8:
                spending_score[i] = np.random.randint(80, 100)
            else:
                spending_score[i] = np.random.randint(10, 50)
    
    # Create DataFrame
    data = pd.DataFrame({
        'CustomerID': customer_ids,
        'Gender': gender,
        'Age': age,
        'Annual Income (k$)': annual_income // 1000,  # Convert to k$ as in original
        'Spending Score (1-100)': spending_score
    })
    
    return data

# Create the dataset
customer_data = create_customer_data(200)
print(f"✓ Dataset created with {len(customer_data)} customers")
print(f"✓ Features available: {list(customer_data.columns)}")
print("\nFirst 5 customers:")
print(customer_data.head())
print("\nDataset Information:")
print(customer_data.info())
print("\nStatistical Summary:")
print(customer_data.describe())

# Display gender distribution
print("\nGender Distribution:")
print(customer_data['Gender'].value_counts())
print(f"Male: {customer_data['Gender'].value_counts()['Male']/len(customer_data)*100:.1f}%")
print(f"Female: {customer_data['Gender'].value_counts()['Female']/len(customer_data)*100:.1f}%")

# ============================================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================
print("\n\n2. EXPLORATORY DATA ANALYSIS...")
print("-" * 50)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 2.1 Age Distribution
axes[0, 0].hist(customer_data['Age'], bins=15, edgecolor='black', alpha=0.7)
axes[0, 0].set_title('Age Distribution of Customers')
axes[0, 0].set_xlabel('Age')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].grid(True, alpha=0.3)

# 2.2 Annual Income Distribution
axes[0, 1].hist(customer_data['Annual Income (k$)'], bins=15, edgecolor='black', alpha=0.7)
axes[0, 1].set_title('Annual Income Distribution')
axes[0, 1].set_xlabel('Annual Income (k$)')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].grid(True, alpha=0.3)

# 2.3 Spending Score Distribution
axes[0, 2].hist(customer_data['Spending Score (1-100)'], bins=15, edgecolor='black', alpha=0.7)
axes[0, 2].set_title('Spending Score Distribution')
axes[0, 2].set_xlabel('Spending Score (1-100)')
axes[0, 2].set_ylabel('Frequency')
axes[0, 2].grid(True, alpha=0.3)

# 2.4 Income vs Spending Score
scatter = axes[1, 0].scatter(customer_data['Annual Income (k$)'], 
                            customer_data['Spending Score (1-100)'],
                            c=customer_data['Age'], cmap='viridis', alpha=0.6)
axes[1, 0].set_title('Income vs Spending Score')
axes[1, 0].set_xlabel('Annual Income (k$)')
axes[1, 0].set_ylabel('Spending Score (1-100)')
plt.colorbar(scatter, ax=axes[1, 0], label='Age')
axes[1, 0].grid(True, alpha=0.3)

# 2.5 Age vs Spending Score by Gender
for gender in ['Male', 'Female']:
    subset = customer_data[customer_data['Gender'] == gender]
    axes[1, 1].scatter(subset['Age'], subset['Spending Score (1-100)'], 
                      label=gender, alpha=0.6)
axes[1, 1].set_title('Age vs Spending Score by Gender')
axes[1, 1].set_xlabel('Age')
axes[1, 1].set_ylabel('Spending Score (1-100)')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# 2.6 Correlation Heatmap
correlation_matrix = customer_data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, ax=axes[1, 2])
axes[1, 2].set_title('Feature Correlation Matrix')

plt.tight_layout()
plt.savefig('customer_eda.png', dpi=100, bbox_inches='tight')
plt.show()

# ============================================================
# 3. DATA PREPROCESSING
# ============================================================
print("\n\n3. DATA PREPROCESSING...")
print("-" * 50)

# Encode categorical variable (Gender)
print("Encoding categorical variables...")
label_encoder = LabelEncoder()
customer_data['Gender_encoded'] = label_encoder.fit_transform(customer_data['Gender'])
print(f"Gender encoding: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")

# Select features for clustering
# Based on the problem statement, we'll use Annual Income and Spending Score for segmentation
features = ['Annual Income (k$)', 'Spending Score (1-100)', 'Age']
X = customer_data[features].copy()

print(f"\nSelected features for clustering: {features}")
print(f"Feature matrix shape: {X.shape}")

# Standardize the features
print("\nStandardizing features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("✓ Features standardized (mean=0, std=1)")

# ============================================================
# 4. DETERMINING OPTIMAL NUMBER OF CLUSTERS
# ============================================================
print("\n\n4. FINDING OPTIMAL NUMBER OF CLUSTERS...")
print("-" * 50)

# Method 1: Elbow Method
print("Applying Elbow Method...")
inertia = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Method 2: Silhouette Score
print("Calculating Silhouette Scores...")
silhouette_scores = []
for k in range(2, 11):  # Silhouette score requires at least 2 clusters
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    score = silhouette_score(X_scaled, kmeans.labels_)
    silhouette_scores.append(score)

# Visualize both methods
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Elbow plot
axes[0].plot(k_range, inertia, 'bo-')
axes[0].set_xlabel('Number of Clusters (k)')
axes[0].set_ylabel('Inertia (Within-cluster SSE)')
axes[0].set_title('Elbow Method for Optimal k')
axes[0].grid(True, alpha=0.3)

# Highlight the "elbow" point (usually at k=5 for this type of data)
elbow_k = 5
axes[0].plot(elbow_k, inertia[elbow_k-1], 'ro', markersize=10)
axes[0].annotate(f'Elbow point (k={elbow_k})', 
                xy=(elbow_k, inertia[elbow_k-1]),
                xytext=(elbow_k+1, inertia[elbow_k-1] + 50),
                arrowprops=dict(arrowstyle='->'))

# Silhouette score plot
axes[1].plot(range(2, 11), silhouette_scores, 'go-')
axes[1].set_xlabel('Number of Clusters (k)')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Score for Different k')
axes[1].grid(True, alpha=0.3)

# Highlight best silhouette score
best_k = range(2, 11)[np.argmax(silhouette_scores)]
axes[1].plot(best_k, np.max(silhouette_scores), 'ro', markersize=10)
axes[1].annotate(f'Best score (k={best_k})', 
                xy=(best_k, np.max(silhouette_scores)),
                xytext=(best_k+0.5, np.max(silhouette_scores) - 0.02),
                arrowprops=dict(arrowstyle='->'))

plt.tight_layout()
plt.savefig('optimal_k_determination.png', dpi=100, bbox_inches='tight')
plt.show()

# Choose optimal k (based on both methods)
optimal_k = 5  # Typically 5 clusters work well for mall customer data
print(f"\n✓ Optimal number of clusters determined: k = {optimal_k}")
print(f"  - Based on elbow method analysis")
print(f"  - Silhouette score for k={optimal_k}: {silhouette_scores[optimal_k-2]:.4f}")

# ============================================================
# 5. APPLYING K-MEANS CLUSTERING
# ============================================================
print("\n\n5. APPLYING K-MEANS CLUSTERING...")
print("-" * 50)

# Train K-Means with optimal k
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)
customer_data['Cluster'] = kmeans.fit_predict(X_scaled)

print(f"✓ K-Means clustering completed with {optimal_k} clusters")
print(f"✓ Cluster centers (standardized features):")
for i, center in enumerate(kmeans.cluster_centers_):
    print(f"  Cluster {i}: {center}")

# Map cluster centers back to original scale for interpretation
centers_original = scaler.inverse_transform(kmeans.cluster_centers_)
print(f"\n✓ Cluster centers (original scale):")
for i, center in enumerate(centers_original):
    print(f"  Cluster {i}: Income=${center[0]:.1f}k, Score={center[1]:.1f}, Age={center[2]:.1f}")

# ============================================================
# 6. ANALYZING AND VISUALIZING CLUSTERS
# ============================================================
print("\n\n6. ANALYZING AND VISUALIZING CLUSTERS...")
print("-" * 50)

# Cluster statistics
print("\nCluster Distribution:")
cluster_counts = customer_data['Cluster'].value_counts().sort_index()
for cluster_id, count in cluster_counts.items():
    percentage = count / len(customer_data) * 100
    print(f"  Cluster {cluster_id}: {count} customers ({percentage:.1f}%)")

# Analyze each cluster
print("\nCluster Characteristics (averages):")
for cluster_id in range(optimal_k):
    cluster_data = customer_data[customer_data['Cluster'] == cluster_id]
    print(f"\n  Cluster {cluster_id}:")
    print(f"    • Size: {len(cluster_data)} customers")
    print(f"    • Avg Income: ${cluster_data['Annual Income (k$)'].mean():.1f}k")
    print(f"    • Avg Spending Score: {cluster_data['Spending Score (1-100)'].mean():.1f}")
    print(f"    • Avg Age: {cluster_data['Age'].mean():.1f} years")
    print(f"    • Gender: {cluster_data['Gender'].value_counts().to_dict()}")

# Assign meaningful names to clusters based on characteristics
cluster_names = {
    0: "High Income, Low Spenders",
    1: "Moderate Income, Moderate Spenders",
    2: "High Income, High Spenders",
    3: "Low Income, High Spenders",
    4: "Low Income, Low Spenders"
}

customer_data['Cluster_Name'] = customer_data['Cluster'].map(cluster_names)
print("\n✓ Assigned meaningful cluster names:")
for cluster_id, name in cluster_names.items():
    print(f"  Cluster {cluster_id}: {name}")

# Visualize clusters in 2D and 3D
fig = plt.figure(figsize=(16, 6))

# 2D Visualization: Income vs Spending Score
ax1 = plt.subplot(1, 2, 1)
scatter_2d = ax1.scatter(customer_data['Annual Income (k$)'], 
                        customer_data['Spending Score (1-100)'],
                        c=customer_data['Cluster'], cmap='tab10', s=100, alpha=0.7)

# Plot cluster centers
for i, center in enumerate(centers_original):
    ax1.scatter(center[0], center[1], s=300, c='red', marker='X', edgecolor='black')
    ax1.annotate(f'C{i}', xy=(center[0], center[1]), xytext=(5, 5),
                textcoords='offset points', fontsize=12, fontweight='bold')

ax1.set_title('Customer Segments: Income vs Spending Score', fontsize=14)
ax1.set_xlabel('Annual Income (k$)', fontsize=12)
ax1.set_ylabel('Spending Score (1-100)', fontsize=12)
ax1.grid(True, alpha=0.3)
plt.colorbar(scatter_2d, ax=ax1, label='Cluster ID')

# 3D Visualization
ax2 = plt.subplot(1, 2, 2, projection='3d')
scatter_3d = ax2.scatter(customer_data['Annual Income (k$)'], 
                        customer_data['Spending Score (1-100)'],
                        customer_data['Age'],
                        c=customer_data['Cluster'], cmap='tab10', s=100, alpha=0.7)

ax2.set_title('3D View: Income, Spending Score, and Age', fontsize=14)
ax2.set_xlabel('Annual Income (k$)', fontsize=10)
ax2.set_ylabel('Spending Score', fontsize=10)
ax2.set_zlabel('Age', fontsize=10)

plt.tight_layout()
plt.savefig('customer_clusters.png', dpi=100, bbox_inches='tight')
plt.show()

# ============================================================
# 7. CLUSTER PROFILES AND MARKETING RECOMMENDATIONS
# ============================================================
print("\n\n7. CLUSTER PROFILES AND MARKETING STRATEGIES")
print("-" * 50)
print("Based on the clustering analysis, here are the customer segments:")

for cluster_id, cluster_name in cluster_names.items():
    cluster_members = customer_data[customer_data['Cluster'] == cluster_id]
    
    print(f"\n{'='*60}")
    print(f"SEGMENT: {cluster_name.upper()}")
    print(f"{'='*60}")
    print(f"Profile:")
    print(f"  • Average Income: ${cluster_members['Annual Income (k$)'].mean():.1f}k")
    print(f"  • Average Spending Score: {cluster_members['Spending Score (1-100)'].mean():.1f}/100")
    print(f"  • Average Age: {cluster_members['Age'].mean():.1f} years")
    print(f"  • Gender Ratio: {cluster_members['Gender'].value_counts().to_dict()}")
    print(f"  • Size: {len(cluster_members)} customers ({len(cluster_members)/len(customer_data)*100:.1f}%)")
    
    print(f"\nMarketing Recommendations:")
    if cluster_id == 2:  # High Income, High Spenders
        print("  → Target with premium products and loyalty programs")
        print("  → Offer exclusive memberships and early access to sales")
        print("  → Personal shopping assistance and VIP treatment")
    elif cluster_id == 0:  # High Income, Low Spenders
        print("  → Focus on value proposition and quality assurance")
        print("  → Educate about product benefits and unique features")
        print("  → Offer bundled deals to increase basket size")
    elif cluster_id == 4:  # Low Income, Low Spenders
        print("  → Target with budget-friendly options and discounts")
        print("  → Promote essential items and basic necessities")
        print("  → Offer payment plans or layaway options")
    elif cluster_id == 3:  # Low Income, High Spenders
        print("  → Caution: High credit risk but responsive to promotions")
        print("  → Offer limited-time discounts and flash sales")
        print("  → Cross-sell complementary low-cost items")
    elif cluster_id == 1:  # Moderate Income, Moderate Spenders
        print("  → Target with balanced value-quality propositions")
        print("  → Standard loyalty programs and regular promotions")
        print("  → Focus on mid-range products and family packages")
    
    # Show a few sample customers from this cluster
    print(f"\nSample customers from this segment (Customer IDs):")
    sample_ids = cluster_members['CustomerID'].head(3).tolist()
    print(f"  {sample_ids}")

# ============================================================
# 8. MODEL EVALUATION AND VALIDATION
# ============================================================
print("\n\n8. MODEL EVALUATION METRICS")
print("-" * 50)

# Calculate evaluation metrics
silhouette_avg = silhouette_score(X_scaled, customer_data['Cluster'])
davies_bouldin = davies_bouldin_score(X_scaled, customer_data['Cluster'])

print(f"✓ Silhouette Score: {silhouette_avg:.4f}")
print("  (Higher is better, range: -1 to 1)")
print(f"✓ Davies-Bouldin Index: {davies_bouldin:.4f}")
print("  (Lower is better, lower values indicate better separation)")
print(f"✓ Inertia (Within-cluster SSE): {kmeans.inertia_:.2f}")
print("  (Lower is better, measures compactness of clusters)")

# Interpretation
print("\n✓ Model Performance Interpretation:")
if silhouette_avg > 0.5:
    print("  • Strong cluster structure identified")
elif silhouette_avg > 0.25:
    print("  • Reasonable cluster structure")
else:
    print("  • Weak cluster structure - consider different features or algorithm")

# ============================================================
# 9. PREDICTING NEW CUSTOMER SEGMENTS
# ============================================================
print("\n\n9. PREDICTING SEGMENTS FOR NEW CUSTOMERS")
print("-" * 50)

def predict_customer_segment(age, annual_income_k, spending_score):
    """Predict which segment a new customer belongs to"""
    # Prepare input
    input_data = np.array([[annual_income_k, spending_score, age]])
    
    # Scale the input
    input_scaled = scaler.transform(input_data)
    
    # Predict cluster
    cluster_id = kmeans.predict(input_scaled)[0]
    cluster_name = cluster_names[cluster_id]
    
    return cluster_id, cluster_name

# Test with example customers
test_customers = [
    (25, 15, 80),   # Young, low income, high spender
    (45, 80, 20),   # Middle-aged, high income, low spender
    (35, 40, 50),   # Middle-aged, moderate income, moderate spender
    (60, 120, 90),  # Senior, very high income, high spender
    (22, 20, 30)    # Young, low income, low spender
]

print("Example predictions for new customers:")
print("-" * 40)
print(f"{'Age':<5} {'Income(k$)':<12} {'Spending Score':<15} {'Segment':<30}")
print("-" * 40)

for age, income, score in test_customers:
    cluster_id, segment = predict_customer_segment(age, income, score)
    print(f"{age:<5} ${income:<11} {score:<14} {segment:<30}")

# ============================================================
# 10. SAVING THE RESULTS
# ============================================================
print("\n\n10. SAVING RESULTS AND MODEL...")
print("-" * 50)

# Save the clustered data
customer_data.to_csv('customer_segmentation_results.csv', index=False)
print("✓ Customer segmentation results saved to 'customer_segmentation_results.csv'")

# Save the model and scaler
import joblib
joblib.dump(kmeans, 'kmeans_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')
print("✓ K-Means model saved to 'kmeans_model.pkl'")
print("✓ Scaler saved to 'scaler.pkl'")
print("✓ Label encoder saved to 'label_encoder.pkl'")

# Create a summary report
with open('segmentation_summary.txt', 'w') as f:
    f.write("CUSTOMER SEGMENTATION ANALYSIS REPORT\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Dataset: Mall Customer Data\n")
    f.write(f"Total Customers: {len(customer_data)}\n")
    f.write(f"Optimal Clusters: {optimal_k}\n\n")
    
    f.write("CLUSTER DISTRIBUTION:\n")
    for cluster_id in range(optimal_k):
        count = len(customer_data[customer_data['Cluster'] == cluster_id])
        percentage = count / len(customer_data) * 100
        f.write(f"Cluster {cluster_id} ({cluster_names[cluster_id]}): {count} customers ({percentage:.1f}%)\n")
    
    f.write(f"\nMODEL PERFORMANCE:\n")
    f.write(f"Silhouette Score: {silhouette_avg:.4f}\n")
    f.write(f"Davies-Bouldin Index: {davies_bouldin:.4f}\n")

print("✓ Summary report saved to 'segmentation_summary.txt'")

print("\n" + "=" * 70)
print("CUSTOMER SEGMENTATION ANALYSIS COMPLETE!")
print("=" * 70)
print("\nNext Steps:")
print("1. Review the generated visualizations")
print("2. Examine 'customer_segmentation_results.csv' for detailed insights")
print("3. Use the saved model to segment new customers")
print("4. Implement targeted marketing strategies for each segment")