import numpy as np
import pandas as pd

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

# Generate synthetic examples within each cluster using SMOTE
def smote(cluster, num_synthetic_samples=1, k=5):
    synthetic_examples = []
    for minority_sample in cluster:
        neighbors = []
        for other_sample in cluster:
            if not np.array_equal(minority_sample, other_sample):
                dist = euclidean_distance(minority_sample, other_sample)
                neighbors.append((other_sample, dist))
        neighbors.sort(key=lambda x: x[1])
        neighbors = [neighbor[0] for neighbor in neighbors[:k]]
        for _ in range(num_synthetic_samples):
            neighbor = neighbors[np.random.randint(len(neighbors))]
            alpha = np.random.rand()
            synthetic_example = minority_sample + alpha * (neighbor - minority_sample)
            synthetic_examples.append(synthetic_example)
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

# Generate and print synthetic examples for each cluster using SMOTE
print("\nSynthetic Examples for Each Cluster (SMOTE):")
for i, cluster in enumerate(clusters):
    synthetic_examples = smote(cluster, num_synthetic_samples=1, k=5)
    print(f"Cluster {i+1}: {synthetic_examples}")
