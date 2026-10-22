import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Configuração da página
st.set_page_config(
    page_title="Dashboard de Manutenção Preditiva 3D",
    page_icon="🔧",
    layout="wide"
)

st.title("🔧 Dashboard de Manutenção Preditiva — Visualização 3D")
st.markdown("""
Dashboard interativo com **DBSCAN** para detecção de anomalias em sensores de chão de fábrica.  
Ajuste os parâmetros na barra lateral e explore o gráfico 3D (Frequência × Temperatura).
""")

# ============================================
# 1. Geração do conjunto de dados com outliers
# ============================================
@st.cache_data
def gerar_dados(n_maquinas=200, seed=42):
    np.random.seed(seed)
    
    # Cluster normal 1
    n1 = int(n_maquinas * 0.45)
    freq1 = np.random.normal(loc=50, scale=3, size=n1)
    temp1 = np.random.normal(loc=65, scale=4, size=n1)
    
    # Cluster normal 2
    n2 = int(n_maquinas * 0.40)
    freq2 = np.random.normal(loc=70, scale=4, size=n2)
    temp2 = np.random.normal(loc=80, scale=5, size=n2)
    
    # Outliers intencionais
    n_out = n_maquinas - n1 - n2
    freq_out = np.random.uniform(20, 100, size=n_out)
    temp_out = np.random.uniform(40, 120, size=n_out)
    
    frequencia = np.concatenate([freq1, freq2, freq_out])
    temperatura = np.concatenate([temp1, temp2, temp_out])
    
    maquina_id = [f"Maq_{i:03d}" for i in range(1, n_maquinas + 1)]
    
    df = pd.DataFrame({
        "Maquina_ID": maquina_id,
        "Frequencia_Vibracao": frequencia,
        "Pico_Temperatura": temperatura
    })
    return df

df = gerar_dados()

# ============================================
# 2. Sidebar - Controles interativos
# ============================================
st.sidebar.header("⚙️ Parâmetros do DBSCAN")

eps = st.sidebar.slider(
    "eps (raio de vizinhança)",
    min_value=0.1,
    max_value=2.0,
    value=0.5,
    step=0.05,
    help="Distância máxima entre dois pontos para serem considerados vizinhos."
)

min_samples = st.sidebar.slider(
    "min_samples",
    min_value=2,
    max_value=20,
    value=5,
    step=1,
    help="Número mínimo de pontos em uma vizinhança para formar um cluster."
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Dicas:**
- `eps` menor → mais clusters e mais outliers  
- `min_samples` maior → clusters mais densos e mais outliers
""")

# ============================================
# 3. Aplicação do DBSCAN
# ============================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[["Frequencia_Vibracao", "Pico_Temperatura"]])

dbscan = DBSCAN(eps=eps, min_samples=min_samples)
labels = dbscan.fit_predict(X_scaled)

df["Cluster"] = labels
df["Anomalia"] = df["Cluster"] == -1
df["Status"] = df["Anomalia"].map({True: "Anomalia", False: "Normal"})

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_anomalias = int((labels == -1).sum())

col1, col2, col3 = st.columns(3)
col1.metric("Total de Máquinas", len(df))
col2.metric("Clusters Detectados", n_clusters)
col3.metric("Anomalias Detectadas", n_anomalias)

# ============================================
# 4. Gráfico 3D interativo (Plotly) — sem legenda
# ============================================
st.subheader("📊 Gráfico 3D — Frequência × Temperatura")

# Cores: clusters coloridos + outliers em preto
unique_clusters = sorted([c for c in df["Cluster"].unique() if c != -1])
# Paleta simples para clusters
palette = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
]
color_map = {c: palette[i % len(palette)] for i, c in enumerate(unique_clusters)}
color_map[-1] = "#000000"  # outliers pretos

fig = go.Figure()

# Adicionar cada cluster (e outliers) como um trace separado para controle de cor
for cluster_id in sorted(df["Cluster"].unique()):
    mask = df["Cluster"] == cluster_id
    subset = df[mask]
    
    is_outlier = cluster_id == -1
    marker_symbol = "x" if is_outlier else "circle"
    marker_size = 10 if is_outlier else 7
    opacity = 0.95 if is_outlier else 0.8
    
    fig.add_trace(go.Scatter3d(
        x=subset["Frequencia_Vibracao"],
        y=subset["Pico_Temperatura"],
        z=subset["Cluster"],          # eixo Z = rótulo do cluster (outliers em -1)
        mode="markers",
        marker=dict(
            size=marker_size,
            color=color_map[cluster_id],
            opacity=opacity,
            symbol=marker_symbol,
            line=dict(width=1, color="white") if not is_outlier else dict(width=0)
        ),
        text=subset["Maquina_ID"],
        customdata=np.stack([
            subset["Maquina_ID"],
            subset["Frequencia_Vibracao"].round(2),
            subset["Pico_Temperatura"].round(2),
            subset["Cluster"]
        ], axis=-1),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>" +
            "Frequência: %{customdata[1]} Hz<br>" +
            "Temperatura: %{customdata[2]} °C<br>" +
            "Cluster: %{customdata[3]}<br>" +
            "<extra></extra>"
        ),
        showlegend=False
    ))

fig.update_layout(
    scene=dict(
        xaxis_title="Frequência de Vibração (Hz)",
        yaxis_title="Pico de Temperatura (°C)",
        zaxis_title="Cluster",
        xaxis=dict(backgroundcolor="rgb(230, 230, 230)", gridcolor="white"),
        yaxis=dict(backgroundcolor="rgb(230, 230, 230)", gridcolor="white"),
        zaxis=dict(backgroundcolor="rgb(230, 230, 230)", gridcolor="white"),
    ),
    margin=dict(l=0, r=0, b=0, t=30),
    height=650,
    title=dict(text="Clusters e Anomalias (DBSCAN) — Outliers em preto (marcador X)", x=0.5),
    showlegend=False
)

st.plotly_chart(fig, use_container_width=True)

# ============================================
# 5. Tabela filtrável (Normais / Anomalias / Todos)
# ============================================
st.subheader("📋 Tabela de Máquinas (filtrável)")

filtro = st.selectbox(
    "Filtrar por status:",
    options=["Todos", "Apenas Anomalias", "Apenas Normais"],
    index=0
)

if filtro == "Apenas Anomalias":
    df_tabela = df[df["Anomalia"]].copy()
elif filtro == "Apenas Normais":
    df_tabela = df[~df["Anomalia"]].copy()
else:
    df_tabela = df.copy()

# Preparar colunas amigáveis
df_exibir = df_tabela[[
    "Maquina_ID",
    "Frequencia_Vibracao",
    "Pico_Temperatura",
    "Cluster",
    "Status"
]].rename(columns={
    "Maquina_ID": "ID da Máquina",
    "Frequencia_Vibracao": "Frequência de Vibração (Hz)",
    "Pico_Temperatura": "Pico de Temperatura (°C)",
    "Cluster": "Cluster",
    "Status": "Status"
}).reset_index(drop=True)

st.dataframe(
    df_exibir.style.format({
        "Frequência de Vibração (Hz)": "{:.2f}",
        "Pico de Temperatura (°C)": "{:.2f}"
    }),
    use_container_width=True,
    hide_index=True,
    height=400
)

# Download
csv = df_exibir.to_csv(index=False).encode("utf-8")
st.download_button(
    label="📥 Baixar tabela filtrada (CSV)",
    data=csv,
    file_name="maquinas_filtradas.csv",
    mime="text/csv"
)

# ============================================
# Rodapé
# ============================================
st.markdown("---")
st.caption("""
**Como funciona:**  
Dados sintéticos gerados com NumPy (dois regimes normais + outliers intencionais).  
DBSCAN identifica regiões densas (clusters) e marca pontos isolados como anomalias (Cluster = -1).  
O gráfico 3D usa o eixo Z como o número do cluster para facilitar a separação visual.  
Passe o mouse sobre qualquer ponto para ver o ID da máquina e os valores de frequência e temperatura.
""")
