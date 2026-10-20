# KDD for Production Loss Analysis (Laminated Biscuit Line)

Python script that applies the **KDD (Knowledge Discovery in Databases)** process to production data from a food manufacturing line, built for my Production Engineering undergraduate thesis (UCS, 2025): *"Aplicação do processo KDD para estudo de processos produtivos na indústria alimentícia"*.

The goal is to understand what drives **relative loss** (loss in kg ÷ production in kg) using process variables that come from a MES (Manufacturing Execution System).

> **All data in this repository is synthetic.** It was randomly generated to demonstrate the code, with the same table structure as the original study. The real data belongs to a company and is not public. Metrics printed by the script on this sample say nothing about the real process.

## KDD steps in the code

| KDD step | What the script does |
|---|---|
| **Selection** | Reads five tables from Excel: production, losses, overweight, fermentation and stoppages |
| **Pre-processing** | Parses dates, converts types, fills missing values |
| **Integration** | Aggregates each table by order/day/shift and merges them into one dataset |
| **Transformation** | Builds relative indicators: loss, overweight, non-conforming packs and stoppage time per category, all normalized by production |
| **Data mining** | Compares Linear Regression, Random Forest and XGBoost on a 70/30 split |
| **Interpretation** | Real vs. predicted plot, plus coefficients (linear) and feature importance (tree models) |

## Data format

`data/laminados_exemplo.xlsx` has one sheet per table:

| Sheet | Main columns used |
|---|---|
| `Produção` | `DIA`, `TURNO`, `OP`, `PESO_TOTAL` |
| `Perdas` | `DIA`, `TURNO`, `ORDEM_PRODUCAO`, `ITEM_PAI`, `PERDA_KG` |
| `Sobrepeso` | `DIA`, `TURNO`, `ITEM`, `PESO_EXCEDENTE_KG`, `PACOTES_NOK`, `PACOTES_TOTAL` |
| `Fermentação` | `ORDEM_PRODUCAO`, `TEMPO_FERMENTACAO` |
| `Paradas` | `DATA_OCORRENCIA_INICIAL`, `TURNO`, `TOTAL_TEMPO_SEGUNDOS`, `ULTIMA_DESCRICAO` |

## How to run

```bash
pip install -r requirements.txt
python laminados_regressao.py
```

Run it from the repository folder so the relative path `data/laminados_exemplo.xlsx` resolves. The file is also organized in `# %%` cells, so it can be run step by step in VS Code or Spyder.

## Tech stack

Python · Pandas · NumPy · Matplotlib · scikit-learn · XGBoost
