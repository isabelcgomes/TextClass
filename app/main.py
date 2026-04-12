import os
import time
import anthropic
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(
    title="Health Plan Support Classifier",
    description="Classifies support tickets into predefined categories using Claude.",
    version="1.0.0",
)

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

SYSTEM_PROMPT = f"""Você é um classificador de tickets de suporte de um plano de saúde.

Sua tarefa é classificar a mensagem do usuário em EXATAMENTE UMA das categorias abaixo.
Responda APENAS com o nome exato da categoria, sem explicações, sem pontuação extra.

Categorias válidas:
{chr(10).join(f'- {label}' for label in VALID_LABELS)}

Regras importantes:
- Se a mensagem indicar risco de vida imediato ou urgência extrema, classifique como "Estou numa emergência de saúde".
- Se a mensagem descrever sintomas sem urgência imediata, classifique como "Tenho sintomas e preciso de atendimento com profissional de saúde".
- Responda SOMENTE com o texto da categoria, nada mais."""


class ClassifyRequest(BaseModel):
    text: str

    model_config = {
        "json_schema_extra": {
            "example": {"text": "Minha filha está com febre alta e dificuldade para respirar."}
        }
    }


class ClassifyResponse(BaseModel):
    predicted_label: str
    is_valid_label: bool
    latency_ms: float


@app.post("/classify", response_model=ClassifyResponse)
def classify(request: ClassifyRequest):
    if not request.text.strip():
        raise HTTPException(status_code=422, detail="O campo 'text' não pode estar vazio.")

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY não configurada.")

    client = anthropic.Anthropic(api_key=api_key)

    start = time.perf_counter()
    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=64,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": request.text}],
        )
    except anthropic.APIError as e:
        raise HTTPException(status_code=502, detail=f"Erro na API Anthropic: {str(e)}")

    latency_ms = (time.perf_counter() - start) * 1000
    predicted_label = message.content[0].text.strip()

    return ClassifyResponse(
        predicted_label=predicted_label,
        is_valid_label=predicted_label in VALID_LABELS,
        latency_ms=round(latency_ms, 2),
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/labels")
def list_labels():
    return {"labels": VALID_LABELS}
