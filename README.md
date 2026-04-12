# Health Plan Support Classifier

Classificador automático de tickets de suporte de plano de saúde usando LLM (Claude), exposto via FastAPI.

---

## Stack e decisões de design

| Componente | Escolha | Justificativa |
|---|---|---|
| LLM | Claude (Anthropic) | Zero-shot com prompt estruturado supera modelos menores treinados nos 100 exemplos disponíveis |
| Framework | FastAPI | Tipagem nativa com Pydantic, docs automáticas via Swagger, alto desempenho |
| Classificação | Zero-shot com system prompt | Rápido de iterar; o prompt é versionável como código |
| Métricas | Accuracy + Macro F1 + Confusion Matrix | Dataset balanceado → accuracy é representativa; F1 por classe expõe onde o modelo erra |

**Por que não fine-tuning?** Com apenas 100 exemplos (10 por classe), fine-tuning é frágil e propenso a overfitting. O LLM zero-shot com bom prompt performa melhor e é trivial de atualizar.

---

## Estrutura do projeto

```
classifier/
├── app/
│   └── main.py          # FastAPI app (endpoint /classify)
├── scripts/
│   └── evaluate.py      # Avaliação em batch + geração de report
├── data/
│   └── classification_dataset.csv
├── reports/             # Criado automaticamente pelo evaluate.py
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Instalar dependências

```bash
pip install -r requirements.txt
```

### 2. Configurar a chave da API

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

### 3. Subir a API

```bash
uvicorn app.main:app --reload
```

```bash
python app/gradio_ui.py
```


A API estará disponível em `http://localhost:8000`.

Documentação interativa: `http://localhost:8000/docs`

A interface abre em `http://localhost:7860`.
---

## Endpoints

### `POST /classify`

Classifica uma mensagem de suporte.

**Request:**
```json
{ "text": "Minha filha está com febre alta e dificuldade para respirar." }
```

**Response:**
```json
{
  "predicted_label": "Estou numa emergência de saúde",
  "is_valid_label": true,
  "latency_ms": 412.5
}
```

### `GET /labels`

Lista todas as categorias válidas.

### `GET /health`

Healthcheck da API.

---

## Avaliação em batch (Parte 2)

Com a API rodando, execute:

```bash
python scripts/evaluate.py \
    --dataset data/classification_dataset.csv \
    --output  reports/evaluation_report.json \
    --api-url http://localhost:8000
```

O script:
1. Lê cada linha do CSV
2. Chama `POST /classify` para cada mensagem
3. Compara com o ground truth
4. Gera um report em `reports/evaluation_report.json` com:
   - Accuracy, Macro F1, Weighted F1
   - F1, Precision, Recall por classe
   - Matriz de confusão completa
   - Top erros de classificação
   - Estatísticas de latência (mean, P50, P90, P99)

**Exemplo de output no terminal:**

```
=================================================================
  EVALUATION REPORT
=================================================================
  Samples evaluated : 100 / 100
  Accuracy          : 94.00%
  Macro F1          : 93.85%
  Weighted F1       : 93.85%

  LATENCY
  Mean: 387ms  |  P50: 361ms  |  P90: 512ms  |  P99: 743ms

  PER-CLASS F1
  1.00 ████████████████████  [10]  Cancelar plano
  0.95 ███████████████████   [10]  Críticas ou sugestões
  ...

  TOP MISCLASSIFICATIONS
  [2x]  "Tenho sintomas e preciso de atendimento..."
        → "Estou numa emergência de saúde"
=================================================================
```

---

## Métricas escolhidas e justificativa

| Métrica | Por quê |
|---|---|
| **Accuracy** | Dataset balanceado → métrica simples e direta |
| **Macro F1** | Trata todas as classes com igual peso; penaliza classes com baixo desempenho |
| **F1 por classe** | Identifica onde o modelo está sistematicamente errando |
| **Matriz de confusão** | Revela padrões de confusão (ex: sintomas vs emergência) |
| **Latência (P50/P90/P99)** | Essencial para SLA em produção; P99 expõe outliers |

---

## Discussão: rumo à produção

### Observabilidade
- Logar cada request: `text`, `predicted_label`, `is_valid_label`, `latency_ms`, timestamp
- Alertar quando `is_valid_label = false` (o modelo "inventou" uma categoria)
- Dashboard de drift: monitorar distribuição de categorias ao longo do tempo

### Prompt como código
- O system prompt vive no repositório Git — cada mudança é um commit
- Rodar o `evaluate.py` como CI gate: qualquer alteração no prompt deve manter accuracy ≥ threshold

### Alternativas de modelagem
- **Few-shot**: incluir 1–2 exemplos por classe no prompt melhora casos limítrofes
- **Embeddings + classificador linear**: mais rápido e barato, mas requer dados rotulados suficientes
- **Fine-tuning**: viável se o volume de dados crescer para milhares de exemplos

### Confiança e fallback
- Adicionar um campo `confidence` ao response (pedir ao LLM que retorne JSON com `label` + `confidence`)
- Mensagens com `confidence < 0.7` → fila para revisão humana

### Escalabilidade
- Cache por hash do texto para mensagens repetidas
- Rate limiting no endpoint
- Deploy em container (Docker) com health check no `/health`
