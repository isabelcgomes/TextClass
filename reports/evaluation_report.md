# 📊 Evaluation Report

## 📌 Summary

- Samples evaluated: **8 / 10**
- Errors: **2**
- Invalid labels: **2**
- Accuracy: **62.50%**
- Macro F1: **58.33%**
- Weighted F1: **60.42%**

## ⏱️ Latency

- Mean: **3090.1 ms** | P50: **3047.8 ms** | P90: **3867.8 ms** | P99: **4057.9 ms**

## 🧠 Per-class F1

- **Cancelar plano** → 1.00 `████████████████████` (support: 1.0)
- **Falar sobre agendamento de exame** → 0.00 `                    ` (support: 1.0)
- **Falar sobre agendamento de uma consulta com médico especialista** → 0.67 `█████████████       ` (support: 2.0)
- **Inclusão ou exclusão de dependentes** → 0.67 `█████████████       ` (support: 2.0)
- **Quero indicação ou ajuda para encontrar um médico na rede credenciada** → 0.67 `█████████████       ` (support: 1.0)
- **Tenho sintomas e preciso de atendimento com profissional de saúde** → 0.50 `██████████          ` (support: 1.0)

## ⚠️ Top Misclassifications

- **Falar sobre agendamento de exame** → *Tenho sintomas e preciso de atendimento com profissional de saúde* (1x)
- **Falar sobre agendamento de uma consulta com médico especialista** → *Quero indicação ou ajuda para encontrar um médico na rede credenciada* (1x)
- **Inclusão ou exclusão de dependentes** → *Tenho sintomas e preciso de atendimento com profissional de saúde* (1x)