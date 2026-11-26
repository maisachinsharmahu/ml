import math
from collections import Counter
import matplotlib.pyplot as plt

ATTRS = ["Outlook", "Temperature", "Humidity", "Wind"]
rows = [
    ("Sunny", "Hot", "High", "Weak", "No"),
    ("Sunny", "Hot", "High", "Strong", "No"),
    ("Overcast", "Hot", "High", "Weak", "Yes"),
    ("Rain", "Mild", "High", "Weak", "Yes"),
    ("Rain", "Cool", "Normal", "Weak", "Yes"),
    ("Rain", "Cool", "Normal", "Strong", "No"),
    ("Overcast", "Cool", "Normal", "Strong", "Yes"),
    ("Sunny", "Mild", "High", "Weak", "No"),
    ("Sunny", "Cool", "Normal", "Weak", "Yes"),
    ("Rain", "Mild", "Normal", "Weak", "Yes"),
    ("Sunny", "Mild", "Normal", "Strong", "Yes"),
    ("Overcast", "Mild", "High", "Strong", "Yes"),
    ("Overcast", "Hot", "Normal", "Weak", "Yes"),
    ("Rain", "Mild", "High", "Strong", "No"),
]
data = [dict(zip(ATTRS + ["Play"], r)) for r in rows]

def entropy(rs):
    counts = Counter(r["Play"] for r in rs)
    n = len(rs)
    return -sum(c / n * math.log2(c / n) for c in counts.values())

def gain(rs, attr):
    total = entropy(rs)
    for v in {r[attr] for r in rs}:
        part = [r for r in rs if r[attr] == v]
        total -= len(part) / len(rs) * entropy(part)
    return total

def id3(rs, attrs):
    labels = {r["Play"] for r in rs}
    if len(labels) == 1:
        return labels.pop()
    if not attrs:
        return Counter(r["Play"] for r in rs).most_common(1)[0][0]
    best = max(attrs, key=lambda a: gain(rs, a))
    rest = [a for a in attrs if a != best]
    return {best: {v: id3([r for r in rs if r[best] == v], rest)
                   for v in sorted({r[best] for r in rs})}}

def show(tree, indent=0):
    for attr, branches in tree.items():
        for v, sub in branches.items():
            if isinstance(sub, dict):
                print("  " * indent + f"{attr} = {v}")
                show(sub, indent + 1)
            else:
                print("  " * indent + f"{attr} = {v}  ->  Play = {sub}")

def predict(tree, sample):
    while isinstance(tree, dict):
        attr = next(iter(tree))
        tree = tree[attr][sample[attr]]
    return tree

print(f"Entropy of the whole data set = {entropy(data):.4f}")
print("Information gain of each attribute at the root:")
for a in sorted(ATTRS, key=lambda a: -gain(data, a)):
    print(f"   {a:<12} {gain(data, a):.4f}")

tree = id3(data, ATTRS)
print("\nDecision tree built by ID3:")
show(tree)

ok = sum(predict(tree, r) == r["Play"] for r in data)
print(f"\nTraining accuracy : {ok}/{len(data)}")
new = {"Outlook": "Sunny", "Temperature": "Cool", "Humidity": "High",
       "Wind": "Strong"}
print("New day", new, "->", predict(tree, new))

pos, leaf_x = {}, [0]
def layout(node, depth, name):
    if not isinstance(node, dict):
        pos[name] = (leaf_x[0], -depth); leaf_x[0] += 1
        return pos[name][0]
    attr = next(iter(node))
    xs = [layout(sub, depth + 1, name + "/" + v)
          for v, sub in node[attr].items()]
    pos[name] = (sum(xs) / len(xs), -depth)
    return pos[name][0]
layout(tree, 0, "root")

fig, ax = plt.subplots(figsize=(10, 5.5))
def draw(node, name):
    x, y = pos[name]
    if not isinstance(node, dict):
        c = "#8fd18f" if node == "Yes" else "#f2a1a1"
        ax.text(x, y, node, ha="center", va="center",
                bbox=dict(boxstyle="round", fc=c))
        return
    attr = next(iter(node))
    ax.text(x, y, attr, ha="center", va="center", fontsize=11,
            bbox=dict(boxstyle="round", fc="#a9c8f0"))
    for v, sub in node[attr].items():
        cx, cy = pos[name + "/" + v]
        ax.plot([x, cx], [y - 0.1, cy + 0.1], color="gray")
        ax.text((x + cx) / 2, (y + cy) / 2, v, fontsize=9, ha="center",
                bbox=dict(fc="white", ec="none", pad=1))
        draw(sub, name + "/" + v)
draw(tree, "root")
ax.set_xlim(-0.6, leaf_x[0] - 0.4); ax.set_ylim(-3.6, 0.5); ax.axis("off")
ax.set_title("ID3 decision tree for the play-tennis data")
plt.show()
