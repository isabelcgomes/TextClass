import pandas as pd
import os
import json
import time
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import requests

load_dotenv()

client = InferenceClient(
    api_key=os.environ["LLM_TOKEN"]
)

# Carregar dataset
data = pd.read_csv('classification_dataset.csv')

import pandas as pd
import re

# carregar dados
data = pd.read_csv('classification_dataset.csv')

# 1. remover linhas com texto vazio ou nulo
data = data.dropna(subset=['text'])
data = data[data['text'].str.strip() != ""]

# 2. limpar texto
def clean_text(text):
    text = str(text)
    
    # remover quebras de linha
    text = text.replace("\n", " ")
    
    # remover espaços duplicados
    text = re.sub(r"\s+", " ", text)
    
    # remover espaços no início/fim
    text = text.strip()
    
    return text

data['text'] = data['text'].apply(clean_text)

# 3. padronizar labels
data['label'] = data['label'].str.strip()

# 4. remover duplicatas (mesmo texto)
data = data.drop_duplicates(subset=['text'])

# 5. resetar índice
data = data.reset_index(drop=True)

# Labels únicas
labels = list(set(data['label'].dropna()))

# def query(texto):
#     response = client.zero_shot_classification(
#         model="tasksource/ModernBERT-base-nli",
#         messages=[
#             {
#                 "role": "system",
#                 "content": "Você é um classificador de textos que responde apenas em JSON válido."
#             },
#             {
#                 "role": "user",
#                 "content": f"""
# Classifique o texto abaixo em UMA das seguintes categorias:
# {labels}

# Texto: "{texto}"

# Responda APENAS com um JSON válido no formato:
# {{
#   "text": "{texto}",
#   "label": "<uma das categorias>"
# }}
# """
#             }
#         ],
#         max_tokens=100
#     )

#     result = response.choices[0].message.content.strip()
#     return result





API_URL = "https://router.huggingface.co/hf-inference/models/facebook/bart-large-mnli"
headers = {
    "Authorization": f"Bearer {os.environ['LLM_TOKEN']}",
}

def query(payload, retries=1):
    for _ in range(retries):
        response = requests.post(API_URL, headers=headers, json=payload, timeout=(5, 15))
        result = response.json()

        # se não tiver erro, retorna
        if not (isinstance(result, dict) and "error" in result):
            return result

        time.sleep(1)

    return result 

results = []

for i, row in data.iterrows():
    payload = {
        "inputs": row['text'],
        "parameters": {"candidate_labels": labels},
    }

    try:
        parsed = query(payload)

        # erro da API
        if isinstance(parsed, dict) and "error" in parsed:
            print(f"Erro na linha {i}: {parsed['error']}")
            results.append({
                "text": row['text'],
                "predicted_label": None,
                "score": None
            })
            continue

        # caso 1: lista
        if isinstance(parsed, list):
            parsed = parsed[0]

        # caso 2: formato clássico (labels/scores)
        if "labels" in parsed:
            best_label = parsed["labels"][0]
            best_score = parsed["scores"][0]

        # caso 3: formato simplificado (label/score)
        elif "label" in parsed:
            best_label = parsed["label"]
            best_score = parsed["score"]

        # fallback
        else:
            print(f"Formato inesperado na linha {i}: {parsed}")
            best_label = None
            best_score = None

        results.append({
            "text": row['text'],
            "predicted_label": best_label,
            "score": best_score 
        })

    except Exception as e:
        print(f"Erro na linha {i}: {e}")
        print("Resposta recebida:", parsed)
        pass

# Criar DataFrame final
results_df = pd.DataFrame(results)

# Juntar com dataset original (opcional)
final_df = pd.concat([data, results_df], axis=1)

# Salvar resultado
final_df.to_csv("classification_results.csv", index=False)

print("Processamento concluído!")