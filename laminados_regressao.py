# %% [markdown]
# # Perda relativa em linha de biscoitos laminados (KDD + regressão)
#
# Trabalho de conclusão de curso (Engenharia de Produção, UCS, 2025): aplicação das etapas do KDD
# a dados de um sistema MES de uma fábrica de alimentos.
#
# Este script integra cinco tabelas (produção, perdas, sobrepeso, fermentação e paradas), cria o
# indicador de **perda relativa** (perda em kg / produção em kg) e compara três modelos de regressão:
# Regressão Linear, Random Forest e XGBoost.
#
# > **Os dados em `data/laminados_exemplo.xlsx` são sintéticos**, gerados só para demonstrar o código.
# > Os dados reais do estudo não são públicos.

# %%
# 1) Linear Regression (baseline interpretável)

# LINHA DE LAMINADOS – Linear Regression (baseline)
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


def to_dt(s): return pd.to_datetime(s, errors="coerce")


def load_lam2_features(xls_path="data/laminados_exemplo.xlsx"):
    xls = Path(xls_path)
    prod = pd.read_excel(xls, "Produção")
    perdas = pd.read_excel(xls, "Perdas")
    sobrepeso = pd.read_excel(xls, "Sobrepeso")
    ferment = pd.read_excel(xls, "Fermentação")
    paradas = pd.read_excel(xls, "Paradas")

    for df_ in [prod, perdas, sobrepeso, ferment]:
        if "DIA" in df_.columns:
            df_["DIA"] = to_dt(df_["DIA"]).dt.date

    prod["PRODUCAO_KG"] = pd.to_numeric(
        prod.get("PESO_TOTAL", prod.get("PESO", 0)), errors="coerce")
    prod_ag = prod.groupby(["OP", "DIA", "TURNO"], dropna=False).agg(
        PRODUCAO_KG=("PRODUCAO_KG", "sum")).reset_index()

    perdas_ag = (perdas.groupby(["ORDEM_PRODUCAO", "DIA", "TURNO", "ITEM_PAI"], dropna=False)
                       .agg(PERDA_KG=("PERDA_KG", "sum")).reset_index()
                       .rename(columns={"ORDEM_PRODUCAO": "OP"}))

    sobrepeso_ag = sobrepeso.groupby(["ITEM", "DIA", "TURNO"], dropna=False).agg(
        PESO_EXCEDENTE_KG=("PESO_EXCEDENTE_KG", "sum"),
        PACOTES_NOK=("PACOTES_NOK", "sum"),
        PACOTES_TOTAL=("PACOTES_TOTAL", "sum")
    ).reset_index()

    ferm_ag = (ferment.groupby("ORDEM_PRODUCAO", dropna=False)["TEMPO_FERMENTACAO"]
                      .mean().reset_index().rename(columns={"ORDEM_PRODUCAO": "OP"}))

    base = (perdas_ag
            .merge(prod_ag, on=["OP", "DIA", "TURNO"], how="left")
            .merge(sobrepeso_ag, left_on=["ITEM_PAI", "DIA", "TURNO"], right_on=["ITEM", "DIA", "TURNO"], how="left")
            .merge(ferm_ag, on="OP", how="left"))

    for c in ["PERDA_KG", "PESO_EXCEDENTE_KG", "PACOTES_NOK", "PACOTES_TOTAL", "TEMPO_FERMENTACAO", "PRODUCAO_KG"]:
        if c in base.columns:
            base[c] = pd.to_numeric(base[c], errors="coerce").fillna(0)

    base["PERC_NOK"] = np.where(
        base["PACOTES_TOTAL"] > 0, base["PACOTES_NOK"]/base["PACOTES_TOTAL"], 0.0)
    base["PERDA_REL"] = np.where(
        base["PRODUCAO_KG"] > 0, base["PERDA_KG"]/base["PRODUCAO_KG"], 0.0)
    base["EXCED_REL"] = np.where(
        base["PRODUCAO_KG"] > 0, base["PESO_EXCEDENTE_KG"]/base["PRODUCAO_KG"], 0.0)

    # Paradas → normalizar por produção
    paradas["TOTAL_TEMPO_SEGUNDOS"] = pd.to_numeric(
        paradas["TOTAL_TEMPO_SEGUNDOS"], errors="coerce").fillna(0)
    if "DIA" not in paradas.columns:
        if "DATA_OCORRENCIA_INICIAL" in paradas.columns:
            paradas["DIA"] = to_dt(paradas["DATA_OCORRENCIA_INICIAL"]).dt.date
        elif "DATA_PROCESSO" in paradas.columns:
            paradas["DIA"] = to_dt(paradas["DATA_PROCESSO"]).dt.date
        else:
            paradas["DIA"] = pd.NaT

    def norm(s): return (s.fillna("").astype(str)
                         .str.normalize("NFKD").str.encode("ascii", "ignore").str.decode("utf-8")
                         .str.upper())
    desc = norm(paradas["ULTIMA_DESCRICAO"])
    cats = {"MANUTEN": "MANUTEN", "SETUP": "SETUP", "LIMPEZA": "LIMPEZA",
            "FALHA_OPERA": "FALHA OPERA|FALHA OPERAC", "DEMANDA": "DEMANDA", "VACUO": "VACUO"}
    for k, pat in cats.items():
        paradas[f"{k}_SEG"] = np.where(desc.str.contains(
            pat), paradas["TOTAL_TEMPO_SEGUNDOS"], 0)

    agg_cols = [f"{k}_SEG" for k in cats.keys()]
    keys = [k for k in ["ORDEM_PRODUCAO", "OP",
                        "DIA", "TURNO"] if k in paradas.columns]
    if not keys:
        par_ag = paradas.assign(_=1).groupby("_", dropna=False).agg(
            {c: "sum" for c in agg_cols}).reset_index(drop=True)
    else:
        par_ag = paradas.groupby(keys, dropna=False).agg(
            {c: "sum" for c in agg_cols}).reset_index()
    if "ORDEM_PRODUCAO" in par_ag.columns:
        par_ag = par_ag.rename(columns={"ORDEM_PRODUCAO": "OP"})

    merge_keys = [k for k in ["OP", "DIA", "TURNO"]
                  if k in base.columns and k in par_ag.columns]
    base = base.merge(par_ag, on=merge_keys, how="left")

    for k in cats.keys():
        col = f"{k}_SEG"
        if col in base.columns:
            base[f"{col}_REL"] = np.where(
                base["PRODUCAO_KG"] > 0, base[col]/base["PRODUCAO_KG"], 0.0)

    # Feature set (apenas processo)
    feat = ["EXCED_REL", "PERC_NOK", "TEMPO_FERMENTACAO"] + \
           [f"{k}_SEG_REL" for k in cats.keys() if f"{k}_SEG_REL" in base.columns]
    X = base[feat].fillna(0)
    y = base["PERDA_REL"].fillna(0)
    return X, y, feat


# ---- Treino/avaliação/plots
X, y, feat = load_lam2_features()
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30, random_state=42)

lr = LinearRegression().fit(Xtr, ytr)
yp = lr.predict(Xte)

mae = mean_absolute_error(yte, yp)
r2 = r2_score(yte, yp)
print(f"LinearRegression | MAE_rel: {mae:.4f} | R²: {r2:.3f}")

# Gráfico 1: Real vs Previsto
plt.figure(figsize=(6, 6))
plt.scatter(yte, yp, alpha=0.5)
lim = float(max(yte.max(), yp.max())) if len(yte) else 1.0
plt.plot([0, lim], [0, lim], '--')
plt.xlabel("Perda relativa real")
plt.ylabel("Perda relativa prevista")
plt.title("Linha de laminados — Linear Regression: Real vs Previsto")
plt.tight_layout()
plt.show()

# Gráfico 2: Coeficientes (importância/efeito)
coef = pd.Series(lr.coef_, index=feat)
coef = coef.sort_values()
plt.figure(figsize=(8, 6))
plt.barh(coef.index, coef.values)
plt.title("Linha de laminados — Coeficientes (Linear Regression)")
plt.tight_layout()
plt.show()


# %% [markdown]
#
# Princípio: modelo linear que estima uma combinação linear das variáveis de processo.
#
# Bom quando: relações são aproximadamente lineares; ótima interpretabilidade (sinal e magnitude do coeficiente).
#
# Limitações: não captura não linearidades/interações; pode subajustar.
#
# Uso recomendado: baseline interpretável; se R² baixo, use modelos não lineares em seguida.

# %%
# LINHA DE LAMINADOS – RandomForestRegressor

# Reaproveita a função do bloco anterior (execute a célula anterior primeiro ou copie a função)
X, y, feat = load_lam2_features()
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30, random_state=42)

rf = RandomForestRegressor(n_estimators=400, random_state=42)
rf.fit(Xtr, ytr)
yp = rf.predict(Xte)

mae = mean_absolute_error(yte, yp)
r2 = r2_score(yte, yp)
print(f"RandomForest | MAE_rel: {mae:.4f} | R²: {r2:.3f}")

# Gráfico 1: Real vs Previsto
plt.figure(figsize=(6, 6))
plt.scatter(yte, yp, alpha=0.5)
lim = float(max(yte.max(), yp.max())) if len(yte) else 1.0
plt.plot([0, lim], [0, lim], '--')
plt.xlabel("Perda relativa real")
plt.ylabel("Perda relativa prevista")
plt.title("Linha de laminados — RF: Real vs Previsto")
plt.tight_layout()
plt.show()

# Gráfico 2: Importância das variáveis
imp = pd.Series(rf.feature_importances_, index=feat).sort_values()
plt.figure(figsize=(8, 6))
plt.barh(imp.index, imp.values)
plt.title("Linha de laminados — Importância (RandomForest)")
plt.tight_layout()
plt.show()


# %% [markdown]
# Princípio: ensemble de muitas árvores (bagging); captura não linearidade e interações.
#
# Bom quando: relações são complexas; geralmente melhora MAE/R² vs linear.
#
# Limitações: menos interpretável que linear; pode tender à classe/intervalo dominante se houver muito desbalanceamento de situações.
#
# Uso: ótimo “cavalo de batalha” para dado tabular industrial.

# %%
# LINHA DE LAMINADOS – XGBoost Regressor

# Reaproveita a função do primeiro bloco
X, y, feat = load_lam2_features()
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30, random_state=42)

xgb = XGBRegressor(
    n_estimators=500, max_depth=6, learning_rate=0.07,
    subsample=0.9, colsample_bytree=0.9, random_state=42, tree_method="hist"
)
xgb.fit(Xtr, ytr)
yp = xgb.predict(Xte)

mae = mean_absolute_error(yte, yp)
r2 = r2_score(yte, yp)
print(f"XGBRegressor | MAE_rel: {mae:.4f} | R²: {r2:.3f}")

# Gráfico 1: Real vs Previsto
plt.figure(figsize=(6, 6))
plt.scatter(yte, yp, alpha=0.5)
lim = float(max(yte.max(), yp.max())) if len(yte) else 1.0
plt.plot([0, lim], [0, lim], '--')
plt.xlabel("Perda relativa real")
plt.ylabel("Perda relativa prevista")
plt.title("Linha de laminados — XGB: Real vs Previsto")
plt.tight_layout()
plt.show()

# Gráfico 2: Importância das variáveis
imp = pd.Series(xgb.feature_importances_, index=feat).sort_values()
plt.figure(figsize=(8, 6))
plt.barh(imp.index, imp.values)
plt.title("Linha de laminados — Importância (XGBoost)")
plt.tight_layout()
plt.show()


# %% [markdown]
# Princípio: boosting de árvores (cada árvore corrige erros da anterior).
#
# Bom quando: há não linearidades e efeitos de interação; costuma ter melhor acurácia em tabular.
#
# Limitações: tuning sensível; interpretabilidade menor que linear (mas melhor que redes).
#
# Uso: modelo “principal” para explicar quanto e quais condições mais impactam a perda.
