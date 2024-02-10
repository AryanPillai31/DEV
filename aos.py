import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Euclidean distance function
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

# K-means clustering function
def k_means_clustering(data, k=2, max_iters=100):
    # Initialize centroids randomly
    centroids = data[np.random.choice(range(len(data)), k, replace=False)]
    
    # Main loop
    for _ in range(max_iters):
        # Assign each data point to the closest centroid
        clusters = [[] for _ in range(k)]
        for point in data:
            distances = [euclidean_distance(point, centroid) for centroid in centroids]
            cluster_assignment = np.argmin(distances)
            clusters[cluster_assignment].append(point)
        
        # Update centroids
        new_centroids = [np.mean(cluster, axis=0) for cluster in clusters if cluster]
        
        # Check for convergence
        if np.all([np.array_equal(c, nc) for c, nc in zip(centroids, new_centroids)]):
            break
        
        centroids = new_centroids
    
    return np.array(centroids), [np.array(cluster) for cluster in clusters]

# Generate synthetic examples based on the selected representatives
def generate_synthetic_examples(centroids, clusters):
    synthetic_examples = []
    for cluster in clusters:
        centroid = np.mean(cluster, axis=0)
        synthetic_examples.append(2 * centroid - cluster)
    return synthetic_examples

# Path to the metadata
metadata_path = r'C:\Users\aryan\OneDrive\Desktop\SkinC\HAM10000_metadata.csv'

# Load metadata from the specified path
metadata = pd.read_csv(metadata_path)

# Encode 'sex' column as numerical values
metadata['sex'] = metadata['sex'].map({'male': 0, 'female': 1})

# Feature selection and preprocessing
features = ['age', 'sex']  # Using 'age' and 'sex' as features for clustering

# Normalize the 'age' column
metadata['age'] = (metadata['age'] - metadata['age'].mean()) / metadata['age'].std()

# Drop rows with missing values
metadata.dropna(subset=features, inplace=True)

# Convert DataFrame to NumPy array
metadata_features = metadata[features].values

# Call the k-means clustering function
num_clusters = 5
centroids, clusters = k_means_clustering(metadata_features, num_clusters)

# Assign cluster labels to the DataFrame
cluster_labels = np.zeros(len(metadata_features))
for i, cluster in enumerate(clusters):
    for point in cluster:
        cluster_labels[np.where((metadata_features == point).all(axis=1))] = i

metadata['cluster'] = cluster_labels.astype(int)

# Print the number of clusters
print("Number of clusters:", len(clusters))

# Print the number of entries in each cluster
for i, cluster in enumerate(clusters):
    print(f"Cluster {i+1} contains {len(cluster)} entries")

# Print the first 100 entries with their assigned clusters
print("\nFirst 100 entries with their assigned clusters:")
print(metadata.head(100))

# Generate synthetic examples within each cluster
synthetic_examples = generate_synthetic_examples(centroids, clusters)

# Print the synthetic examples
for i, synthetic_cluster in enumerate(synthetic_examples):
    print(f"\nSynthetic Examples for Cluster {i+1}:")
    print(synthetic_cluster)
