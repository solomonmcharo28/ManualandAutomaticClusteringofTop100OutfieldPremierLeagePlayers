import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# -----------------------------
# 1. Load Excel file
# -----------------------------
file_path = "Book2.xlsx"
df = pd.read_excel(file_path)

# -----------------------------
# 2. Select required columns
# -----------------------------
cols = ["age", "weight_kg", "mcharo_fender_ln_price", "short_name"]
df = df[cols].dropna()

# -----------------------------
# 3. Create 3D plot
# -----------------------------
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

ax.scatter(
    df["age"],
    df["weight_kg"],
    df["mcharo_fender_ln_price"],
    alpha=0.6
)

# -----------------------------
# 4. Label points (IMPORTANT PART)
# -----------------------------
for _, row in df.iterrows():
    ax.text(
        row["age"],
        row["weight_kg"],
        row["mcharo_fender_ln_price"],
        row["short_name"],
        fontsize=7
    )

ax.set_xlabel("Age")
ax.set_ylabel("Weight (kg)")
ax.set_zlabel("Mcharo-Fender ln(Price)")
ax.set_title("Age vs Weight vs Mcharo-Fender ln(Price)")
plt.savefig("primerarecard.png")
plt.show()
