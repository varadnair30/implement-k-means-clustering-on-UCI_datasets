# Varad Nair | 1002161475

try:    
    import sys
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    import warnings


    

    def random_seed_func(feats, k, rs):
        if rs==1:
            rs=None
        n = feats.shape[0]
        np.random.seed(rs)
        clust_ids = np.random.randint(0, k, n)
        centroids = np.array([feats[clust_ids == i].mean(axis=0) for i in range(k)])
        return centroids

    

    def euclidean_dist_func(c1, c2):
        return np.sqrt(np.sum((c1 - c2) ** 2, axis=-1))

    

    def k_means_func(feats, k, random_seed):
        if random_seed is not None:
            np.random.seed(random_seed)
        centroids = random_seed_func(feats, k, random_seed)
        itrs=0
        while itrs < 20:
            dists = euclidean_dist_func(feats[:, np.newaxis], centroids)
            clust_ids = np.argmin(dists, axis=1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                centroids = np.array([feats[clust_ids == i].mean(axis=0) for i in range(k)])

                
            itrs += 1
        return clust_ids, centroids


    

    def sse_err_func(feats, centroids, clust_ids):
        dists = euclidean_dist_func(feats, centroids[clust_ids])
        return np.sum(dists)


    


    if __name__ == "__main__":
        
        # Replace with the random seed value of your choice.
        random_seed = 2
        if len(sys.argv) < 2:
            print("Incorrect Input")
            sys.exit(1)  
        file_path = sys.argv[1]
        file_data = np.loadtxt(file_path)
        file_name = os.path.basename(file_path)
        feats = file_data[:, :-1]
        sse_errs = []
        k_vals = list(range(2, 11))
        print("")
        print(file_name)
        print("")
        for k in k_vals:
            try:
                clust_ids, centroids = k_means_func(feats, k, random_seed)
                err=sse_err_func(feats, centroids, clust_ids)
                sse_errs.append(err)
                print(f"For k = {k} After 20 iterations: SSE Error = {err:.4f}")
            except RuntimeWarning:
                pass
        
        
        
        
        plt.plot(k_vals, sse_errs, marker='o')
        plt.xlabel("K")
        plt.ylabel("SSE")
        plt.title("GRAPH OF SSE VS. K -> Number of Clusters")
        plt.grid(True)
        plt.show()

except RuntimeWarning:
    pass
