import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()
X, y, names = iris.data, iris.target, iris.target_names
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                          stratify=y, random_state=42)

def knn_predict(x, k):
    d = np.sqrt(((X_tr - x) ** 2).sum(axis=1))
    nearest = y_tr[np.argsort(d)[:k]]
    return Counter(nearest).most_common(1)[0][0]

print(f"Training samples = {len(X_tr)}, test samples = {len(X_te)}")
accs = []
for k in range(1, 16):
    pred = np.array([knn_predict(x, k) for x in X_te])
    accs.append((pred == y_te).mean())
    if k in (1, 3, 5, 7, 9, 15):
        print(f"  k = {k:<2}  accuracy = {accs[-1] * 100:.2f} %")
best_k = int(np.argmax(accs)) + 1
print(f"Best k = {best_k}")

k = 5
pred = np.array([knn_predict(x, k) for x in X_te])
print(f"\nConfusion matrix for k = {k}")
cm = confusion_matrix(y_te, pred)
print(cm)

sk = KNeighborsClassifier(n_neighbors=k).fit(X_tr, y_tr)
print("Same predictions as scikit-learn :",
      (sk.predict(X_te) == pred).all())

flower = np.array([5.9, 3.0, 5.1, 1.8])
print("\nNew flower (sepal 5.9 x 3.0, petal 5.1 x 1.8) is classified as:",
      names[knn_predict(flower, k)])

fig, ax = plt.subplots(1, 3, figsize=(14, 4))
ax[0].plot(range(1, 16), np.array(accs) * 100, "o-")
ax[0].set_xlabel("k"); ax[0].set_ylabel("accuracy (%)")
ax[0].set_title("Accuracy for different k")
ax[1].imshow(cm, cmap="Blues")
for i in range(3):
    for j in range(3):
        ax[1].text(j, i, cm[i, j], ha="center", va="center")
ax[1].set_xticks(range(3)); ax[1].set_xticklabels(names, rotation=20)
ax[1].set_yticks(range(3)); ax[1].set_yticklabels(names)
ax[1].set_title(f"Confusion matrix (k = {k})")
for c in range(3):
    ax[2].scatter(X[y == c, 2], X[y == c, 3], label=names[c], s=18)
ax[2].scatter(flower[2], flower[3], marker="*", s=300, c="red",
              edgecolor="k", label="new flower")
ax[2].set_xlabel("petal length"); ax[2].set_ylabel("petal width")
ax[2].legend(); ax[2].set_title("Iris flowers")
plt.tight_layout()
plt.show()
