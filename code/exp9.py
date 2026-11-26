import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

rng = np.random.default_rng(2)
X, _ = make_blobs(n_samples=120, centers=3, cluster_std=0.9, random_state=6)
X = np.vstack([X, [[14, 14], [15, 13], [13.5, 15]]])
K = 3

def pam(X, k, seed=0):
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
    r = np.random.default_rng(seed)
    med = list(r.choice(len(X), k, replace=False))
    cost = D[:, med].min(axis=1).sum()
    swaps = 0
    while True:
        best = (cost, None, None)
        for mi in range(k):
            for cand in range(len(X)):
                if cand in med:
                    continue
                trial = med.copy(); trial[mi] = cand
                c = D[:, trial].min(axis=1).sum()
                if c < best[0]:
                    best = (c, mi, cand)
        if best[1] is None:
            break
        med[best[1]] = best[2]; cost = best[0]; swaps += 1
    return D[:, med].argmin(axis=1), np.array(med), cost, swaps

labels, med, cost, swaps = pam(X, K)
print(f"Points = {len(X)} (3 of them are outliers), k = {K}")
print(f"PAM finished after {swaps} swaps, total cost = {cost:.2f}")
print("Medoids (actual data points):")
print(np.round(X[med], 2))
print("Cluster sizes :", np.bincount(labels))

km = KMeans(n_clusters=K, n_init=10, random_state=0).fit(X)
print("\nK-Means centroids (means, not data points):")
print(np.round(km.cluster_centers_, 2))
print("Cluster sizes :", np.bincount(km.labels_))
out_cluster = labels[-1]
print("\nAll three outliers are in the same PAM cluster :",
      len(set(labels[-3:])) == 1)

fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
ax[0].scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=18)
ax[0].scatter(X[med, 0], X[med, 1], marker="X", s=220, c="red",
              edgecolor="k", label="medoids")
ax[0].set_title("K-Medoids (PAM)"); ax[0].legend()
ax[1].scatter(X[:, 0], X[:, 1], c=km.labels_, cmap="viridis", s=18)
ax[1].scatter(*km.cluster_centers_.T, marker="X", s=220, c="red",
              edgecolor="k", label="centroids")
ax[1].set_title("K-Means with the same outliers"); ax[1].legend()
plt.tight_layout()
plt.show()
