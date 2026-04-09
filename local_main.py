import pandas as pd
import re
from transformers import pipeline

# =========================
# 1. Carregar e limpar dados
# =========================

data = pd.read_csv('classification_dataset.csv')

data = data.dropna(subset=['text'])
data = data[data['text'].str.strip() != ""]

def clean_text(text):
    text = str(text)
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

data['text'] = data['text'].apply(clean_text)
data['label'] = data['label'].str.strip()
data = data.drop_duplicates(subset=['text']).reset_index(drop=True)

# Labels únicas
labels = list(set(data['label']))

# =========================
# 2. Modelo local (CPU)
# =========================

classifier = pipeline(
    "zero-shot-classification",
    model="alt_model", 
    tokenizer="alt_model",
    device=-1,  # 🔥 força CPU
    local_files_only=True
)

# =========================
# 3. Classificação
# =========================

results = []

for i, row in data.iterrows():
    print(f"Processando linha {i}...")

    try:
        result = classifier(
            row['text'],
            candidate_labels=labels
        )

        best_label = result['labels'][0]
        best_score = result['scores'][0]

        results.append({
            "predicted_label": best_label,
            "score": best_score
        })

    except Exception as e:
        print(f"Erro na linha {i}: {e}")
        results.append({
            "predicted_label": None,
            "score": None
        })

# =========================
# 4. Salvar resultado
# =========================

results_df = pd.DataFrame(results)
final_df = pd.concat([data, results_df], axis=1)

final_df.to_csv("classification_results.csv", index=False)

print("Processamento concluído!")