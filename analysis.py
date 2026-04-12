import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

# =========================
# LOAD
# =========================
df = pd.read_csv("classification_results.csv")
df["correct"] = df["label"] == df["predicted_label"]
df["text_length"] = df["text"].astype(str).apply(len)

accuracy = accuracy_score(df["label"], df["predicted_label"])

# =========================
# STYLE PADRÃO
# =========================
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "axes.titlesize": 16,
    "axes.labelsize": 12
})

# =========================
# 1. OVERVIEW
# =========================
plt.figure()
sns.countplot(x="correct", data=df)
plt.title(f"Model Overview (Accuracy: {accuracy:.1%})")
plt.xticks([0, 1], ["Error", "Correct"])
plt.xlabel("")
plt.ylabel("Number of samples")
plt.tight_layout()
plt.savefig("01_overview.png", dpi=300)
plt.close()

# =========================
# 2. CONFUSION MATRIX
# =========================
plt.figure()

labels = sorted(df["label"].unique())
cm = confusion_matrix(df["label"], df["predicted_label"], labels=labels)
cm_norm = cm / cm.sum(axis=1, keepdims=True)

sns.heatmap(
    cm_norm,
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels,
    cbar_kws={"label": "Proportion"}
)

plt.title("Confusion Matrix (Normalized)")
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("02_confusion_matrix.png", dpi=300)
plt.close()

# =========================
# 3. ACCURACY BY CLASS
# =========================
plt.figure()

acc_by_class = df.groupby("label")["correct"].mean().sort_values()

sns.barplot(
    x=acc_by_class.values,
    y=acc_by_class.index
)

plt.title("Accuracy by Class")
plt.xlabel("Accuracy")
plt.ylabel("Class")
plt.tight_layout()
plt.savefig("03_accuracy_by_class.png", dpi=300)
plt.close()

# =========================
# 4. TOP ERRORS
# =========================
plt.figure()

errors = df[df["correct"] == False]

top_errors = (
    errors.groupby(["label", "predicted_label"])
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
    .head(10)
)

top_errors["error_pair"] = (
    top_errors["label"] + " → " + top_errors["predicted_label"]
)

sns.barplot(
    data=top_errors,
    y="error_pair",
    x="count"
)

plt.title("Most Frequent Misclassifications")
plt.xlabel("Count")
plt.ylabel("")
plt.tight_layout()
plt.savefig("04_top_errors.png", dpi=300)
plt.close()

# =========================
# 5. TEXT LENGTH ANALYSIS
# =========================
plt.figure()

sns.kdeplot(
    data=df,
    x="text_length",
    hue="correct",
    fill=True,
    common_norm=False
)

plt.title("Text Length Distribution (Correct vs Error)")
plt.xlabel("Text length")
plt.ylabel("Density")
plt.legend(title="Prediction", labels=["Error", "Correct"])
plt.tight_layout()
plt.savefig("05_text_length_analysis.png", dpi=300)
plt.close()

print("Todas as visualizações foram geradas com sucesso!")