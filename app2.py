"""
Dashboard de Manutenção Preditiva (3D)
Executar com:  streamlit run dashboard_manutencao_preditiva.py
Dependências:  pip install streamlit numpy pandas scikit-learn seaborn plotly
"""

import numpy as np
import pandas as pd
import seaborn as sns
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Manutenção Preditiva", page_icon="🏭", layout="wide")


# ----------------------------------------------------------------------
# 1. Geração dos dados (NumPy)
# ----------------------------------------------------------------------
@st.cache_data
def gerar_dados(n_normais: int, n_outliers: int, semente: int) -> pd.DataFrame:
    rng = np.random.default_rng(semente)

    # Dois regimes de operação normal
    n1 = n_normais // 2
    n2 = n_normais - n1
    regime_a = np.column_stack([
        rng.normal(50, 3, n1),     # Frequencia_Vibracao (Hz)
        rng.normal(70, 4, n1),     # Pico_Temperatura (°C)
    ])
    regime_b = np.column_stack([
        rng.normal(80, 4, n2),
        rng.normal(95, 5, n2),
    ])

    # Outliers intencionais (falhas / desgaste)
    outliers = np.column_stack([
        rng.uniform(20, 120, n_outliers),
        rng.uniform(30, 150, n_outliers),
    ])

    dados = np.vstack([regime_a, regime_b, outliers])
    rng.shuffle(dados)

    df = pd.DataFrame(dados, columns=["Frequencia_Vibracao", "Pico_Temperatura"])
    df.insert(0, "Maquina", [f"M-{i:03d}" for i in range(1, len(df) + 1)])
    return df.round(2)


# ----------------------------------------------------------------------
# 2. Interface – controles laterais
# ----------------------------------------------------------------------
st.title("🏭 Dashboard de Manutenção Preditiva")
st.caption("Detecção de anomalias em sensores de chão de fábrica com DBSCAN")

with st.sidebar:
    st.header("⚙️ Parâmetros do DBSCAN")
    eps = st.slider("eps (raio de vizinhança)", 0.05, 1.50, 0.30, 0.05)
    min_samples = st.slider("min_samples", 2, 30, 5, 1)

    st.header("📊 Dados simulados")
    n_normais = st.slider("Máquinas em operação normal", 50, 400, 150, 10)
    n_outliers = st.slider("Outliers intencionais", 0, 40, 12, 1)
    semente = st.number_input("Semente aleatória", value=42, step=1)

    st.header("🎨 Visualização")
    mostrar_rotulos = st.checkbox("Mostrar valores em cada máquina", value=True)

# ----------------------------------------------------------------------
# 3. DBSCAN (com padronização, pois as escalas das métricas diferem)
# ----------------------------------------------------------------------
df = gerar_dados(n_normais, n_outliers, int(semente)).copy()

X = StandardScaler().fit_transform(df[["Frequencia_Vibracao", "Pico_Temperatura"]])
df["Cluster"] = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
df["Status"] = np.where(df["Cluster"] == -1, "Anomalia", "Normal")

n_clusters = df.loc[df["Cluster"] != -1, "Cluster"].nunique()
n_anomalias = int((df["Status"] == "Anomalia").sum())

c1, c2, c3 = st.columns(3)
c1.metric("Total de máquinas", len(df))
c2.metric("Clusters normais", n_clusters)
c3.metric("Anomalias", n_anomalias)

# ----------------------------------------------------------------------
# 4. Gráfico 3D: Frequência x Temperatura x Máquina (sem legenda)
# ----------------------------------------------------------------------
st.subheader("Frequência de Vibração × Pico de Temperatura (3D)")

paleta = sns.color_palette("tab10", max(n_clusters, 1)).as_hex()
df["Eixo_Maquina"] = df["Maquina"].str.extract(r"(\d+)").astype(int)  # eixo Z = nº da máquina

fig = go.Figure()

for cluster in sorted(df["Cluster"].unique()):
    sub = df[df["Cluster"] == cluster]
    anomalo = cluster == -1
    cor = "black" if anomalo else paleta[cluster % len(paleta)]

    fig.add_trace(go.Scatter3d(
        x=sub["Frequencia_Vibracao"],
        y=sub["Pico_Temperatura"],
        z=sub["Eixo_Maquina"],
        mode="markers+text" if mostrar_rotulos else "markers",
        text=[f"{f:.1f} Hz | {t:.1f}°C"
              for f, t in zip(sub["Frequencia_Vibracao"], sub["Pico_Temperatura"])],
        textposition="top center",
        textfont=dict(size=8, color=cor),
        marker=dict(
            size=7 if anomalo else 5,
            color=cor,
            symbol="diamond" if anomalo else "circle",
            line=dict(width=1, color="white"),
            opacity=0.95,
        ),
        customdata=np.column_stack([sub["Maquina"], sub["Cluster"]]),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Vibração: %{x:.2f} Hz<br>"
            "Temperatura: %{y:.2f} °C<br>"
            "Cluster: %{customdata[1]}<extra></extra>"
        ),
        showlegend=False,
    ))

fig.update_layout(
    showlegend=False,  # sem legenda
    height=700,
    margin=dict(l=0, r=0, t=10, b=0),
    scene=dict(
        xaxis_title="Frequencia_Vibracao (Hz)",
        yaxis_title="Pico_Temperatura (°C)",
        zaxis_title="Nº da máquina",
    ),
)
st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------
# 5. Tabela interativa com filtro Normais / Anomalias
# ----------------------------------------------------------------------
st.subheader("📋 Tabela de máquinas")

filtro = st.radio(
    "Filtrar por status:",
    ["Todas", "Normal", "Anomalia"],
    horizontal=True,
)

tabela = df[["Maquina", "Frequencia_Vibracao", "Pico_Temperatura", "Cluster", "Status"]]
if filtro != "Todas":
    tabela = tabela[tabela["Status"] == filtro]


def destacar(linha):
    cor = "background-color: #ffd6d6" if linha["Status"] == "Anomalia" else ""
    return [cor] * len(linha)


st.dataframe(
    tabela.style.apply(destacar, axis=1),
    use_container_width=True,
    hide_index=True,
)
st.caption(f"Exibindo {len(tabela)} de {len(df)} máquinas.")
