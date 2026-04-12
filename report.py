import pandas as pd

# =========================
# LOAD
# =========================
df = pd.read_csv("classification_results.csv")
df["correct"] = df["label"] == df["predicted_label"]

# =========================
# TOP ERROS
# =========================
errors = df[df["correct"] == False]

top_errors = (
    errors.groupby(["label", "predicted_label"])
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
    .head(10)
)

total_errors = len(errors)

# =========================
# BUILD MARKDOWN
# =========================
md = []
md.append("# 🚨 Top 10 Erros de Classificação\n")

md.append(f"Total de erros: **{total_errors}**\n")

md.append("| Ranking | Classe Real | Classe Predita | Quantidade | % dos Erros |")
md.append("|---------|-------------|----------------|------------|-------------|")

for i, row in top_errors.iterrows():
    pct = (row["count"] / total_errors) * 100
    md.append(
        f"| {i+1} | {row['label']} | {row['predicted_label']} | {row['count']} | {pct:.1f}% |"
    )

# =========================
# INSIGHT AUTOMÁTICO
# =========================
top_3_pct = top_errors.head(3)["count"].sum() / total_errors * 100

md.append("\n---\n")
md.append(
    f"**Insight:** Os 3 erros mais frequentes representam **{top_3_pct:.1f}%** de todos os erros de classificação."
)

# =========================
# SAVE
# =========================
with open("relatorio_erros.md", "w", encoding="utf-8") as f:
    f.write("\n".join(md))

print("Relatório salvo como relatorio_erros.md")