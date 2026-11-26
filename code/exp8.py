import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import adjusted_rand_score

X, y_true = make_moons(n_samples=300, noise=0.07, random_state=3)
X = np.vstack([X, [[2.0, 1.5], [-1.5, 1.5], [2.2, -1.0]]])

def dbscan(X, eps, min_pts):
    n = len(X)
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
    neigh = [np.where(D[i] <= eps)[0] for i in range(n)]
    labels = np.full(n, -1)
    cluster = 0
    for i in range(n):
        if labels[i] != -1 or len(neigh[i]) < min_pts:
            continue
        labels[i] = cluster
        queue = list(neigh[i])
        while queue:
            j = queue.pop()
            if labels[j] == -1:
                labels[j] = cluster
                if len(neigh[j]) >= min_pts:
                    queue.extend(neigh[j])
        cluster += 1
    return labels

labels = dbscan(X, eps=0.18, min_pts=5)
print(f"Points = {len(X)}")
print("Clusters found :", len(set(labels) - {-1}))
print("Noise points   :", int((labels == -1).sum()))
ids = sorted(set(labels) - {-1})
print("Cluster sizes  :", [int((labels == c).sum()) for c in ids])

sk = DBSCAN(eps=0.18, min_samples=5).fit_predict(X)
print("Agreement with scikit-learn (ARI) :",
      round(adjusted_rand_score(sk, labels), 3))

km = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(X)
print(f"K-Means agreement with the two moons : "
      f"{adjusted_rand_score(y_true, km[:300]):.3f}")
print(f"DBSCAN  agreement with the two moons : "
      f"{adjusted_rand_score(y_true, labels[:300]):.3f}")

print("\nEffect of eps (min_pts = 5)")
for eps in (0.05, 0.1, 0.18, 0.4, 0.8):
    lab = dbscan(X, eps, 5)
    print(f"  eps = {eps:<4} clusters = {len(set(lab) - {-1})}, "
          f"noise = {(lab == -1).sum()}")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
noise = labels == -1
ax[0].scatter(X[~noise, 0], X[~noise, 1], c=labels[~noise],
              cmap="viridis", s=18)
ax[0].scatter(X[noise, 0], X[noise, 1], c="red", marker="x", s=60,
              label="noise")
ax[0].set_title("DBSCAN (eps = 0.18, min_pts = 5)"); ax[0].legend()
ax[1].scatter(X[:, 0], X[:, 1], c=km, cmap="viridis", s=18)
ax[1].set_title("K-Means (k = 2) on the same data")
plt.tight_layout()
plt.show()
