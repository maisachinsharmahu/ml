import numpy as np
import pandas as pd

a = np.array([[1, 2, 3], [4, 5, 6]])
print("Array a:\n", a)
print("shape =", a.shape, " dtype =", a.dtype)
print("a * 2 =\n", a * 2)
print("column means :", a.mean(axis=0))
print("a @ a.T =\n", a @ a.T)
np.arange(0, 10, 2)

marks = pd.DataFrame({
    "student": ["Aman", "Bhavya", "Chirag", "Divya", "Esha", "Farhan"],
    "branch":  ["CSE", "IT", "CSE", "IT", "CSE", "IT"],
    "maths":   [78, 85, 62, 90, 71, 55],
    "physics": [66, 80, 70, 88, 64, 59],
})
marks["total"] = marks["maths"] + marks["physics"]
marks

marks[["maths", "physics", "total"]].describe()

print(marks.groupby("branch")[["maths", "physics", "total"]].mean())
marks.sort_values("total", ascending=False).head(3)

%config InlineBackend.figure_format = "retina"
import matplotlib.pyplot as plt
marks.plot(x="student", y=["maths", "physics"], kind="bar",
           figsize=(7, 3.2))
plt.title("Marks of the students")
plt.ylabel("marks")
plt.show()
