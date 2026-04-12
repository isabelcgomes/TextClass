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
import tempfile

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

import matplotlib.pyplot as plt

import hashlib

# 🔧 patch para erro do reportlab no Windows
_original_md5 = hashlib.md5

def md5_compat(*args, **kwargs):
    kwargs.pop("usedforsecurity", None)
    return _original_md5(*args, **kwargs)

hashlib.md5 = md5_compat


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

def generate_charts(df):
    import tempfile

    latencies = pd.to_numeric(df["Latência (ms)"], errors="coerce").dropna()

    counts = (
        df[~df["Categoria"].str.startswith("⚠️")]["Categoria"]
        .value_counts()
    )

    # 🔽 gráfico 1: distribuição de categorias
    fig1 = plt.figure()
    counts.plot(kind="barh")
    plt.xlabel("Quantidade")
    plt.ylabel("Categoria")
    plt.title("Distribuição por categoria")
    plt.tight_layout()

    chart1 = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    plt.savefig(chart1.name)
    plt.close(fig1)

    # 🔽 gráfico 2: histograma de latência
    fig2 = plt.figure()
    plt.hist(latencies, bins=10)
    plt.xlabel("Latência (ms)")
    plt.ylabel("Frequência")
    plt.title("Distribuição de latência")
    plt.tight_layout()

    chart2 = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    plt.savefig(chart2.name)
    plt.close(fig2)

    return chart1.name, chart2.name

def generate_pdf_report(df, elapsed, total, errors):
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    import tempfile

    styles = getSampleStyleSheet()

    latencies = pd.to_numeric(df["Latência (ms)"], errors="coerce").dropna()
    avg_latency = latencies.mean() if len(latencies) > 0 else 0

    counts = (
        df[~df["Categoria"].str.startswith("⚠️")]["Categoria"]
        .value_counts()
        .reset_index()
    )
    counts.columns = ["Categoria", "Qtd"]

    # 🔽 gerar gráficos
    chart1_path, chart2_path = generate_charts(df)

    pdf_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    doc = SimpleDocTemplate(pdf_file.name)

    elements = []

    # 🔽 título
    elements.append(Paragraph("Relatório de Classificação", styles["Title"]))
    elements.append(Spacer(1, 12))

    # 🔽 resumo
    elements.append(Paragraph(f"Total de mensagens: {total}", styles["Normal"]))
    elements.append(Paragraph(f"Classificadas: {total - errors}", styles["Normal"]))
    elements.append(Paragraph(f"Erros: {errors}", styles["Normal"]))
    elements.append(Paragraph(f"Latência média: {avg_latency:.1f} ms", styles["Normal"]))
    elements.append(Paragraph(f"Tempo total: {elapsed:.2f} s", styles["Normal"]))
    elements.append(Spacer(1, 16))

    # 🔽 gráfico 1
    elements.append(Paragraph("Distribuição por categoria", styles["Heading2"]))
    elements.append(Image(chart1_path, width=400, height=200))
    elements.append(Spacer(1, 16))

    # 🔽 gráfico 2
    elements.append(Paragraph("Distribuição de latência", styles["Heading2"]))
    elements.append(Image(chart2_path, width=400, height=200))
    elements.append(Spacer(1, 16))

    # 🔽 tabela de distribuição
    elements.append(Paragraph("Tabela de categorias", styles["Heading2"]))
    table_data = [["Categoria", "Qtd"]] + counts.values.tolist()

    table = Table(table_data)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
    ]))

    elements.append(table)
    elements.append(Spacer(1, 16))

    # 🔽 tabela detalhada (amostra)
    # elements.append(Paragraph("Resultados (amostra)", styles["Heading2"]))
    # sample_df = df.head(20)
    # table_data = [sample_df.columns.tolist()] + sample_df.values.tolist()

    # table = Table(table_data)
    # table.setStyle(TableStyle([
    #     ("GRID", (0, 0), (-1, -1), 0.25, colors.black),
    # ]))

    # elements.append(table)

    doc.build(elements)

    return pdf_file.name

def classify_csv(file, progress=gr.Progress(track_tqdm=True)):
    if file is None:
        return None, "⚠️ Nenhum arquivo enviado.", "", None, None

    # --- Load CSV ---
    try:
        df = pd.read_csv(file.name)
    except Exception as e:
        return None, f"❌ Erro ao ler o CSV: {e}", "", None

    if "text" not in df.columns:
        return (
            None,
            "❌ O CSV precisa ter uma coluna chamada **`text`**.",
            "",
            None,
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

    # 🔽 NOVO: salvar CSV temporário
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    result_df.to_csv(tmp_file.name, index=False, encoding="utf-8-sig")

    # --- Summary ---
    success = total - errors
    summary = (
        f"✅ **{success}/{total}** mensagens classificadas "
        f"em **{elapsed:.1f}s** &nbsp;·&nbsp; "
        f"{'⚠️ ' + str(errors) + ' erros' if errors else '0 erros'}"
    )

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

    # 🔽 retorna também o arquivo
    pdf_path = generate_pdf_report(result_df, elapsed, total, errors)
    return result_df, summary, dist_md, tmp_file.name, pdf_path


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
            download_file = gr.File(label="📥 Baixar resultados em CSV")
            download_pdf = gr.File(label="📄 Baixar relatório em PDF")

    classify_btn.click(
    fn=classify_csv,
    inputs=[file_input],
    outputs=[results_out, summary_out, stats_out, download_file, download_pdf],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_api=False)
