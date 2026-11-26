import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

iris = load_iris()
X, y, names = iris.data, iris.target, iris.target_names

Xs = (X - X.mean(axis=0)) / X.std(axis=0)
cov = np.cov(Xs, rowvar=False)
vals, vecs = np.linalg.eigh(cov)
order = np.argsort(vals)[::-1]
vals, vecs = vals[order], vecs[:, order]
ratio = vals / vals.sum()
Z_pca = Xs @ vecs[:, :2]

print("Eigenvalues           :", np.round(vals, 4))
print("Explained variance (%):", np.round(ratio * 100, 2))
print("Cumulative (%)        :", np.round(np.cumsum(ratio) * 100, 2))

sk = PCA(n_components=2).fit(Xs)
print("\nscikit-learn PCA ratio:",
      np.round(sk.explained_variance_ratio_, 4))
same = np.allclose(np.abs(sk.transform(Xs)), np.abs(Z_pca))
print("Same components as scikit-learn (up to sign) :", same)

lda = LinearDiscriminantAnalysis(n_components=2).fit(X, y)
Z_lda = lda.transform(X)
print("\nLDA explained variance ratio :",
      np.round(lda.explained_variance_ratio_, 4))
print(f"LDA classification accuracy  : {lda.score(X, y) * 100:.2f} %")

fig, ax = plt.subplots(1, 3, figsize=(14, 4))
ax[0].bar(range(1, 5), ratio * 100, color="#4c72b0")
ax[0].plot(range(1, 5), np.cumsum(ratio) * 100, "ro-")
ax[0].set_xlabel("principal component"); ax[0].set_ylabel("variance (%)")
ax[0].set_title("PCA scree plot")
for c in range(3):
    ax[1].scatter(Z_pca[y == c, 0], Z_pca[y == c, 1], label=names[c], s=18)
    ax[2].scatter(Z_lda[y == c, 0], Z_lda[y == c, 1], label=names[c], s=18)
ax[1].set_xlabel("PC1"); ax[1].set_ylabel("PC2")
ax[1].set_title("PCA (unsupervised)")
ax[2].set_xlabel("LD1"); ax[2].set_ylabel("LD2")
ax[2].set_title("LDA (supervised)")
ax[1].legend()
plt.tight_layout()
plt.show()
