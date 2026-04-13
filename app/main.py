import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI(
    title="Health Plan Support Classifier",
    description="Classifies support tickets using Hugging Face zero-shot model.",
    version="1.0.0",
)

# 🔽 labels
VALID_LABELS = [
    "Cancelar plano",
    "Críticas ou sugestões",
    "Entender valores do boleto",
    "Estou numa emergência de saúde",
    "Falar sobre agendamento de exame",
    "Falar sobre agendamento de uma consulta com médico especialista",
    "Inclusão ou exclusão de dependentes",
    "Quero indicação ou ajuda para encontrar um médico na rede credenciada",
    "Quero tirar dúvida sobre reembolso",
    "Tenho sintomas e preciso de atendimento com profissional de saúde",
]

# 🔽 carregar modelo (uma vez só)
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
)


class ClassifyRequest(BaseModel):
    text: str


class ClassifyResponse(BaseModel):
    predicted_label: str
    predicted_score: float
    is_valid_label: bool
    latency_ms: float


@app.post("/classify", response_model=ClassifyResponse)
def classify(request: ClassifyRequest):
    if not request.text.strip():
        raise HTTPException(status_code=422, detail="O campo 'text' não pode estar vazio.")

    start = time.perf_counter()

    try:
        result = classifier(
            request.text,
            candidate_labels=VALID_LABELS,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    latency_ms = (time.perf_counter() - start) * 1000

    predicted_label = result["labels"][0]
    predicted_score = result["scores"][0]

    return ClassifyResponse(
        predicted_label=predicted_label,
        predicted_score=round(predicted_score, 4),
        is_valid_label=predicted_label in VALID_LABELS,
        latency_ms=round(latency_ms, 2),
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/labels")
def list_labels():
    return {"labels": VALID_LABELS}