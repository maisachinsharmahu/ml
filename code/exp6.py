import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.naive_bayes import GaussianNB

iris = load_iris()
X, y, names = iris.data, iris.target, iris.target_names
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                          stratify=y, random_state=1)
classes = np.unique(y_tr)

prior = np.array([(y_tr == c).mean() for c in classes])
mean = np.array([X_tr[y_tr == c].mean(axis=0) for c in classes])
var = np.array([X_tr[y_tr == c].var(axis=0) for c in classes]) + 1e-9

def log_posterior(x):
    ll = -0.5 * np.log(2 * np.pi * var) - (x - mean) ** 2 / (2 * var)
    return np.log(prior) + ll.sum(axis=1)

def predict(x):
    return np.argmax(log_posterior(x))

for n, p in zip(names, prior):
    print(f"Prior P({n}) = {p:.3f}")
print("Mean of petal length per class :", np.round(mean[:, 2], 3))
pred = np.array([predict(x) for x in X_te])
print(f"\nAccuracy on {len(X_te)} test flowers = "
      f"{accuracy_score(y_te, pred) * 100:.2f} %")
cm = confusion_matrix(y_te, pred)
print(cm)

sk = GaussianNB().fit(X_tr, y_tr)
print("Agreement with scikit-learn GaussianNB :",
      f"{(sk.predict(X_te) == pred).mean() * 100:.1f} %")

flower = np.array([6.3, 2.8, 5.0, 1.7])
lp = log_posterior(flower)
post = np.exp(lp - lp.max()); post /= post.sum()
print("\nPosterior probabilities for a new flower", flower)
for n, p in zip(names, post):
    print(f"   P({n:<10} | x) = {p:.4f}")
print("Predicted class :", names[np.argmax(post)])

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
xs = np.linspace(0, 8, 300)
for c in classes:
    g = np.exp(-(xs - mean[c, 2]) ** 2 / (2 * var[c, 2]))
    ax[0].plot(xs, g / np.sqrt(2 * np.pi * var[c, 2]), label=names[c])
ax[0].set_xlabel("petal length"); ax[0].set_ylabel("density")
ax[0].set_title("Class-wise Gaussian of petal length"); ax[0].legend()
ax[1].imshow(cm, cmap="Greens")
for i in range(3):
    for j in range(3):
        ax[1].text(j, i, cm[i, j], ha="center", va="center")
ax[1].set_xticks(range(3)); ax[1].set_xticklabels(names, rotation=20)
ax[1].set_yticks(range(3)); ax[1].set_yticklabels(names)
ax[1].set_title("Confusion matrix")
plt.tight_layout()
plt.show()
