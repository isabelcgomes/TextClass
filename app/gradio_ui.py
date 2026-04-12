"""
gradio_ui.py — Interface Gradio para o classificador de tickets de suporte.

Uso:
    python app/gradio_ui.py

Por padrão, conecta na API em http://localhost:8000.
Para outro endereço: API_URL=http://... python app/gradio_ui.py
"""

import os
import time

import gradio as gr
import pandas as pd
import requests

API_URL = os.getenv("API_URL", "http://localhost:8000")

LABEL_EMOJI = {
    "Cancelar plano": "❌",
    "Críticas ou sugestões": "💬",
    "Entender valores do boleto": "💰",
    "Estou numa emergência de saúde": "🚨",
    "Falar sobre agendamento de exame": "🔬",
    "Falar sobre agendamento de uma consulta com médico especialista": "👨‍⚕️",
    "Inclusão ou exclusão de dependentes": "👨‍👩‍👧",
    "Quero indicação ou ajuda para encontrar um médico na rede credenciada": "📍",
    "Quero tirar dúvida sobre reembolso": "💳",
    "Tenho sintomas e preciso de atendimento com profissional de saúde": "🤒",
}


def classify_csv(file, progress=gr.Progress(track_tqdm=True)):
    if file is None:
        return None, "⚠️ Nenhum arquivo enviado.", ""

    # --- Load CSV ---
    try:
        df = pd.read_csv(file.name)
    except Exception as e:
        return None, f"❌ Erro ao ler o CSV: {e}", ""

    if "text" not in df.columns:
        return (
            None,
            "❌ O CSV precisa ter uma coluna chamada **`text`**.",
            "",
        )

    texts = df["text"].fillna("").tolist()
    total = len(texts)

    results = []
    errors = 0
    t0 = time.perf_counter()

    for i, text in enumerate(progress.tqdm(texts, desc="Classificando...")):
        try:
            resp = requests.post(
                f"{API_URL}/classify",
                json={"text": text},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            label = data["predicted_label"]
            latency = data["latency_ms"]
            emoji = LABEL_EMOJI.get(label, "🏷️")
            results.append(
                {
                    "Texto original": text,
                    "Categoria": f"{emoji} {label}",
                    "Latência (ms)": f"{latency:.0f}",
                }
            )
        except Exception as e:
            errors += 1
            results.append(
                {
                    "Texto original": text,
                    "Categoria": f"⚠️ Erro: {e}",
                    "Latência (ms)": "—",
                }
            )

    elapsed = time.perf_counter() - t0
    result_df = pd.DataFrame(results)

    # --- Summary ---
    success = total - errors
    summary = (
        f"✅ **{success}/{total}** mensagens classificadas "
        f"em **{elapsed:.1f}s** &nbsp;·&nbsp; "
        f"{'⚠️ ' + str(errors) + ' erros' if errors else '0 erros'}"
    )

    # Category distribution for the stats box
    if success > 0:
        counts = (
            result_df[~result_df["Categoria"].str.startswith("⚠️")]["Categoria"]
            .value_counts()
            .reset_index()
        )
        counts.columns = ["Categoria", "Qtd"]
        dist_md = "### Distribuição das categorias\n\n"
        dist_md += "| Categoria | Qtd |\n|---|---|\n"
        for _, row in counts.iterrows():
            dist_md += f"| {row['Categoria']} | {row['Qtd']} |\n"
    else:
        dist_md = ""

    return result_df, summary, dist_md


# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@400;500&display=swap');

:root {
    --bg: #0d0f12;
    --surface: #14181e;
    --surface2: #1c2128;
    --border: #2a3140;
    --accent: #00e5a0;
    --accent2: #0099ff;
    --warn: #ff4f4f;
    --text: #e8ecf0;
    --muted: #6b7888;
    --radius: 10px;
}

body, .gradio-container {
    background: var(--bg) !important;
    font-family: 'Syne', sans-serif !important;
    color: var(--text) !important;
}

/* Header */
#header {
    text-align: center;
    padding: 40px 20px 24px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 32px;
}
#header h1 {
    font-size: 2rem;
    font-weight: 800;
    color: var(--text);
    letter-spacing: -0.03em;
    margin: 0 0 6px;
}
#header h1 span { color: var(--accent); }
#header p {
    color: var(--muted);
    font-size: 0.9rem;
    font-family: 'DM Mono', monospace;
    margin: 0;
}

/* Panels */
.panel {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 20px !important;
}

/* Upload area */
.upload-zone {
    border: 2px dashed var(--border) !important;
    border-radius: var(--radius) !important;
    background: var(--surface2) !important;
    transition: border-color 0.2s;
}
.upload-zone:hover { border-color: var(--accent) !important; }

/* Classify button */
#classify-btn {
    background: var(--accent) !important;
    color: #0d0f12 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    border: none !important;
    border-radius: var(--radius) !important;
    padding: 14px !important;
    cursor: pointer !important;
    transition: opacity 0.2s, transform 0.1s;
    width: 100% !important;
}
#classify-btn:hover { opacity: 0.88 !important; transform: translateY(-1px); }
#classify-btn:active { transform: translateY(0); }

/* Summary */
#summary {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 12px 18px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.85rem !important;
    color: var(--accent) !important;
    min-height: 0 !important;
}

/* Stats panel */
#stats {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 16px !important;
    font-size: 0.85rem !important;
    color: var(--text) !important;
}

/* Results table */
#results-table {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    font-size: 0.85rem !important;
}
#results-table table { width: 100% !important; border-collapse: collapse !important; }
#results-table th {
    background: var(--surface2) !important;
    color: var(--muted) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    padding: 10px 14px !important;
    border-bottom: 1px solid var(--border) !important;
    text-align: left !important;
}
#results-table td {
    padding: 10px 14px !important;
    border-bottom: 1px solid var(--border) !important;
    vertical-align: top !important;
    color: var(--text) !important;
}
#results-table tr:last-child td { border-bottom: none !important; }
#results-table tr:hover td { background: var(--surface2) !important; }

/* Labels */
label, .label-wrap { color: var(--muted) !important; font-size: 0.78rem !important; font-family: 'DM Mono', monospace !important; }
"""

with gr.Blocks(css=CSS, title="Classificador de Tickets") as demo:

    gr.HTML("""
    <div id="header">
        <h1>Ticket <span>Classifier</span></h1>
        <p>health plan support · llm-powered · zero-shot</p>
    </div>
    """)

    with gr.Row():
        # ── Left column: inputs ──────────────────────────────
        with gr.Column(scale=1, min_width=280):
            gr.HTML("<p style='color:#6b7888;font-size:0.78rem;font-family:DM Mono,monospace;margin:0 0 8px;'>ENTRADA</p>")
            file_input = gr.File(
                label="Arquivo CSV (coluna `text` obrigatória)",
                file_types=[".csv"],
                elem_classes=["upload-zone"],
            )
            classify_btn = gr.Button(
                "▶  Classificar",
                elem_id="classify-btn",
                variant="primary",
            )
            summary_out = gr.Markdown(
                value="",
                elem_id="summary",
            )
            stats_out = gr.Markdown(
                value="",
                elem_id="stats",
            )

        # ── Right column: results ────────────────────────────
        with gr.Column(scale=3):
            gr.HTML("<p style='color:#6b7888;font-size:0.78rem;font-family:DM Mono,monospace;margin:0 0 8px;'>RESULTADOS</p>")
            results_out = gr.DataFrame(
                headers=["Texto original", "Categoria", "Latência (ms)"],
                datatype=["str", "str", "str"],
                interactive=False,
                wrap=True,
                elem_id="results-table",
            )

    classify_btn.click(
        fn=classify_csv,
        inputs=[file_input],
        outputs=[results_out, summary_out, stats_out],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_api=False)
