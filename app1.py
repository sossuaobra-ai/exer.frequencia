import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Configuração da página
st.set_page_config(
    page_title="Dashboard de Manutenção Preditiva",
    page_icon="🔧",
    layout="wide"
)

st.title("🔧 Dashboard de Manutenção Preditiva")
st.markdown("""
Este dashboard utiliza **DBSCAN** para detectar anomalias em sensores de chão de fábrica.
Ajuste os parâmetros ao lado para explorar diferentes configurações de clustering.
""")

# ============================================
# 1. Geração do conjunto de dados com outliers
# ============================================
@st.cache_data
def gerar_dados(n_maquinas=200, seed=42):
    np.random.seed(seed)
    
    # Cluster normal 1 (funcionamento estável)
    n1 = int(n_maquinas * 0.45)
    freq1 = np.random.normal(loc=50, scale=3, size=n1)
    temp1 = np.random.normal(loc=65, scale=4, size=n1)
    
    # Cluster normal 2 (outro regime de operação)
    n2 = int(n_maquinas * 0.40)
    freq2 = np.random.normal(loc=70, scale=4, size=n2)
    temp2 = np.random.normal(loc=80, scale=5, size=n2)
    
    # Outliers intencionais (anomalias)
    n_out = n_maquinas - n1 - n2
    freq_out = np.random.uniform(20, 100, size=n_out)
    temp_out = np.random.uniform(40, 120, size=n_out)
    
    # Combinar
    frequencia = np.concatenate([freq1, freq2, freq_out])
    temperatura = np.concatenate([temp1, temp2, temp_out])
    
    # IDs das máquinas
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
**Dicas de uso:**
- Valores menores de `eps` → mais clusters e mais outliers
- Valores maiores de `min_samples` → clusters mais densos e mais outliers
""")

# ============================================
# 3. Aplicação do DBSCAN
# ============================================
# Normalização (importante para DBSCAN)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[["Frequencia_Vibracao", "Pico_Temperatura"]])

# DBSCAN
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
labels = dbscan.fit_predict(X_scaled)

df["Cluster"] = labels
df["Anomalia"] = df["Cluster"] == -1

# Estatísticas rápidas
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_anomalias = (labels == -1).sum()

col1, col2, col3 = st.columns(3)
col1.metric("Total de Máquinas", len(df))
col2.metric("Clusters Detectados", n_clusters)
col3.metric("Anomalias Detectadas", n_anomalias, delta=None if n_anomalias == 0 else f"{n_anomalias} outliers")

# ============================================
# 4. Gráfico de dispersão com Seaborn
# ============================================
st.subheader("📊 Visualização dos Clusters e Anomalias")

fig, ax = plt.subplots(figsize=(10, 6))

# Separar pontos normais e outliers
df_normal = df[df["Cluster"] != -1]
df_outlier = df[df["Cluster"] == -1]

# Plotar clusters normais
if not df_normal.empty:
    sns.scatterplot(
        data=df_normal,
        x="Frequencia_Vibracao",
        y="Pico_Temperatura",
        hue="Cluster",
        palette="tab10",
        s=80,
        alpha=0.8,
        ax=ax,
        legend="full"
    )

# Destacar outliers em preto
if not df_outlier.empty:
    sns.scatterplot(
        data=df_outlier,
        x="Frequencia_Vibracao",
        y="Pico_Temperatura",
        color="black",
        s=120,
        marker="X",
        label="Anomalia (Outlier)",
        ax=ax
    )

ax.set_title("Clusters de Funcionamento Normal vs Anomalias (DBSCAN)", fontsize=14, pad=15)
ax.set_xlabel("Frequência de Vibração (Hz)")
ax.set_ylabel("Pico de Temperatura (°C)")
ax.legend(title="Cluster", loc="best")
ax.grid(True, linestyle="--", alpha=0.4)

st.pyplot(fig)

# ============================================
# 5. Tabela interativa de máquinas anômalas
# ============================================
st.subheader("🚨 Máquinas Consideradas Anômalas")

if n_anomalias > 0:
    df_anomalias = df[df["Anomalia"]].copy()
    df_anomalias = df_anomalias[["Maquina_ID", "Frequencia_Vibracao", "Pico_Temperatura"]].reset_index(drop=True)
    df_anomalias.columns = ["ID da Máquina", "Frequência de Vibração", "Pico de Temperatura"]
    
    st.dataframe(
        df_anomalias.style.format({
            "Frequência de Vibração": "{:.2f}",
            "Pico de Temperatura": "{:.2f}"
        }),
        use_container_width=True,
        hide_index=True
    )
    
    st.download_button(
        label="📥 Baixar lista de anomalias (CSV)",
        data=df_anomalias.to_csv(index=False).encode("utf-8"),
        file_name="maquinas_anomalas.csv",
        mime="text/csv"
    )
else:
    st.success("Nenhuma anomalia detectada com os parâmetros atuais. Tente reduzir o `eps` ou aumentar o `min_samples`.")

# ============================================
# Rodapé informativo
# ============================================
st.markdown("---")
st.caption("""
**Como funciona:**  
Os dados foram gerados sinteticamente com NumPy, contendo dois regimes normais de operação e outliers intencionais.  
O DBSCAN identifica regiões densas (clusters) e marca pontos isolados como anomalias (-1).  
A normalização com StandardScaler garante que as duas métricas tenham a mesma escala.
""")
