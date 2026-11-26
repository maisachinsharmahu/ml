import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

hours = np.array([1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5,
                  5, 5.5, 6, 6.5, 7, 7.5, 8])
marks = np.array([34, 40, 43, 49, 51, 57, 56, 64,
                  67, 68, 75, 74, 82, 84, 88])

x_bar, y_bar = hours.mean(), marks.mean()
sxy = np.sum((hours - x_bar) * (marks - y_bar))
sxx = np.sum((hours - x_bar) ** 2)
b1 = sxy / sxx
b0 = y_bar - b1 * x_bar
pred = b0 + b1 * hours

ss_res = np.sum((marks - pred) ** 2)
ss_tot = np.sum((marks - y_bar) ** 2)
r2 = 1 - ss_res / ss_tot
rmse = np.sqrt(ss_res / len(marks))

print(f"Mean of x = {x_bar:.3f}, mean of y = {y_bar:.3f}")
print(f"Slope     b1 = {b1:.4f}")
print(f"Intercept b0 = {b0:.4f}")
print(f"Equation : marks = {b0:.2f} + {b1:.2f} * hours")
print(f"R-squared = {r2:.4f}    RMSE = {rmse:.3f}")

sk = LinearRegression().fit(hours.reshape(-1, 1), marks)
print(f"\nscikit-learn : slope = {sk.coef_[0]:.4f}, "
      f"intercept = {sk.intercept_:.4f}")

new = 6.8
print(f"\nPrediction for {new} hours of study : {b0 + b1 * new:.1f} marks")

plt.figure(figsize=(7, 4.5))
plt.scatter(hours, marks, color="#1f77b4", label="students")
plt.plot(hours, pred, color="red", label=f"y = {b0:.1f} + {b1:.1f}x")
for x, y, p in zip(hours, marks, pred):
    plt.plot([x, x], [y, p], color="gray", lw=0.8)
plt.scatter([new], [b0 + b1 * new], color="green", marker="*", s=180,
            label=f"prediction for {new} h")
plt.xlabel("hours studied"); plt.ylabel("marks")
plt.title("Simple linear regression"); plt.legend(); plt.grid(alpha=0.3)
plt.show()
