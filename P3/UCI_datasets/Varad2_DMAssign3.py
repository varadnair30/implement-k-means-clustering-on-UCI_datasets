# Varad Nair | 1002161475

try:    
    import sys
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    import warnings


    def initialize_centroids(features, num_clusters, r_seed):
        if r_seed == 1:
            r_seed = None
        num_samples = features.shape[0]
        np.random.seed(r_seed)
        cluster_ids = np.random.randint(0, num_clusters, num_samples)
        centroids = np.array([features[cluster_ids == i].mean(axis=0) for i in range(num_clusters)])
        return centroids


    def calculate_euclidean_distance(c1, c2):
        return np.sqrt(np.sum((c1 - c2) ** 2, axis=-1))


    def k_means_algorithm(features, num_clusters, random_seed):
        if random_seed is not None:
            np.random.seed(random_seed)
        centroids = initialize_centroids(features, num_clusters, random_seed)
        iterations = 0
        while iterations < 20:
            distances = calculate_euclidean_distance(features[:, np.newaxis], centroids)
            cluster_ids = np.argmin(distances, axis=1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                centroids = np.array([features[cluster_ids == i].mean(axis=0) for i in range(num_clusters)])
            iterations += 1
        return cluster_ids, centroids


    def calculate_sse(features, centroids, cluster_ids):
        distances = calculate_euclidean_distance(features, centroids[cluster_ids])
        return np.sum(distances)


    if __name__ == "__main__":
        
        # Alter the random seed value.
        random_seed_val = 2
        if len(sys.argv) < 2:
            print("Invalid Input")
            sys.exit(1)  
        input_file_path = sys.argv[1]
        file_data = np.loadtxt(input_file_path)
        file_name = os.path.basename(input_file_path)
        features = file_data[:, :-1]
        sse_errors = []
        cluster_values = list(range(2, 11))
        print("")
        print(file_name)
        print("")
        for num_clusters in cluster_values:
            try:
                cluster_ids, centroids = k_means_algorithm(features, num_clusters, random_seed_val)
                error = calculate_sse(features, centroids, cluster_ids)
                sse_errors.append(error)
                print(f"For k = {num_clusters} After 20 iterations: SSE Error = {error:.4f}")
            except RuntimeWarning:
                pass
        
        plt.plot(cluster_values, sse_errors, marker="o")
        plt.xlabel("K")
        plt.ylabel("SSE")
        plt.title("SSE vs. K-> Number of Clusters")
        plt.grid(True)
        plt.show()

except RuntimeWarning:
    pass
