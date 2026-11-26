import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score,
                             normalized_mutual_info_score,
                             homogeneity_score, completeness_score,
                             silhouette_score)

digits = load_digits()
X, y = digits.data, digits.target
print("Images :", X.shape[0], "  pixels per image :", X.shape[1])

km = KMeans(n_clusters=10, n_init=10, random_state=0).fit(X)
labels = km.labels_
print(f"Inertia (SSE)        : {km.inertia_:.1f}")
print(f"Iterations           : {km.n_iter_}")
print(f"Silhouette score     : {silhouette_score(X, labels):.3f}")
print(f"Adjusted Rand Index  : {adjusted_rand_score(y, labels):.3f}")
nmi = normalized_mutual_info_score(y, labels)
print(f"Normalised mutual info : {nmi:.3f}")
print(f"Homogeneity          : {homogeneity_score(y, labels):.3f}")
print(f"Completeness         : {completeness_score(y, labels):.3f}")

table = np.zeros((10, 10), dtype=int)
for c, d in zip(labels, y):
    table[c, d] += 1
mapping = table.argmax(axis=1)
acc = np.mean(mapping[labels] == y)
print(f"\nAccuracy after naming clusters by majority digit : "
      f"{acc * 100:.2f} %")
print("\nrows = cluster, columns = real digit")
print("      " + " ".join(f"{d:>3}" for d in range(10)))
for c in range(10):
    print(f"C{c:<3}  " + " ".join(f"{v:>3}" for v in table[c]))

fig = plt.figure(figsize=(13, 5))
for c in range(10):
    ax = fig.add_subplot(2, 8, c + 1 if c < 5 else c + 4)
    ax.imshow(km.cluster_centers_[c].reshape(8, 8), cmap="gray_r")
    ax.set_title(f"C{c} -> {mapping[c]}", fontsize=9); ax.axis("off")
ax = fig.add_subplot(1, 2, 2)
Z = PCA(n_components=2).fit_transform(X)
sc = ax.scatter(Z[:, 0], Z[:, 1], c=labels, cmap="tab10", s=8)
ax.set_title("Clusters shown on the first two principal components")
ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
fig.suptitle("Cluster centres (as images) and cluster map", y=1.0)
plt.tight_layout()
plt.show()
