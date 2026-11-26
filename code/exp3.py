import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

hours = np.array([0.5, 0.75, 1, 1.25, 1.5, 1.75, 1.75, 2, 2.25, 2.5,
                  2.75, 3, 3.25, 3.5, 4, 4.25, 4.5, 4.75, 5, 5.5])
passed = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
                   1, 0, 1, 0, 1, 1, 1, 1, 1, 1])

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

w, b, lr = 0.0, 0.0, 0.1
losses = []
for _ in range(20000):
    p = sigmoid(w * hours + b)
    loss = -np.mean(passed * np.log(p) + (1 - passed) * np.log(1 - p))
    losses.append(loss)
    w -= lr * np.mean((p - passed) * hours)
    b -= lr * np.mean(p - passed)

pred = (sigmoid(w * hours + b) >= 0.5).astype(int)
print(f"Learned weight w = {w:.4f}, bias b = {b:.4f}")
print(f"Final log-loss   = {losses[-1]:.4f}")
print(f"Decision boundary (p = 0.5) at {-b / w:.2f} hours")
print(f"Training accuracy = {accuracy_score(passed, pred) * 100:.1f} %")
print("Confusion matrix [[TN FP] [FN TP]]:")
print(confusion_matrix(passed, pred))

sk = LogisticRegression(C=1e6).fit(hours.reshape(-1, 1), passed)
print(f"\nscikit-learn : w = {sk.coef_[0][0]:.4f}, "
      f"b = {sk.intercept_[0]:.4f}")
for h in (1.5, 3.0, 4.5):
    print(f"P(pass | {h} hours) = {sigmoid(w * h + b):.3f}")

xs = np.linspace(0, 6, 200)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].scatter(hours, passed, c=passed, cmap="coolwarm", edgecolor="k")
ax[0].plot(xs, sigmoid(w * xs + b), color="black")
ax[0].axhline(0.5, color="gray", ls="--")
ax[0].axvline(-b / w, color="green", ls="--", label="boundary")
ax[0].set_xlabel("hours studied"); ax[0].set_ylabel("P(pass)")
ax[0].set_title("Logistic (sigmoid) curve"); ax[0].legend()
ax[1].plot(losses)
ax[1].set_xlabel("iteration"); ax[1].set_ylabel("log-loss")
ax[1].set_title("Training loss")
plt.tight_layout()
plt.show()
