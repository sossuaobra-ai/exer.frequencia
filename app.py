import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Configuração inicial da página do Streamlit
st.set_page_config(
    page_title="Dashboard de Manutenção Preditiva",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ Dashboard de Manutenção Preditiva - Detecção de Anomalias")
st.markdown("""
Este aplicativo simula dados de sensores de chão de fábrica (**Frequência de Vibração** e **Pico de Temperatura**) 
e utiliza o algoritmo **DBSCAN** para identificar comportamentos normais e detectar possíveis falhas (outliers).
""")

# -----------------------------------------------------------------------------
# 1. GERAÇÃO DO CONJUNTO DE DADOS (NumPy / Pandas)
# -----------------------------------------------------------------------------
@st.cache_data
def gerar_dados(n_samples=300, seed=42):
    np.random.seed(seed)
    
    # Cluster 1: Funcionamento Normal - Carga Leve
    vib_1 = np.random.normal(loc=30, scale=4, size=int(n_samples * 0.5))
    temp_1 = np.random.normal(loc=60, scale=3, size=int(n_samples * 0.5))
    
    # Cluster 2: Funcionamento Normal - Carga Alta
    vib_2 = np.random.normal(loc=65, scale=5, size=int(n_samples * 0.4))
    temp_2 = np.random.normal(loc=85, scale=4, size=int(n_samples * 0.4))
    
    # Outliers (Valores discrepantes / Anomalias intencionais)
    n_outliers = int(n_samples * 0.1)
    vib_out = np.random.uniform(10, 100, size=n_outliers)
    temp_out = np.random.uniform(40, 120, size=n_outliers)
    
    # União dos dados
    vibracao = np.concatenate([vib_1, vib_2, vib_out])
    temperatura = np.concatenate([temp_1, temp_2, temp_out])
    
    # Criar IDs para as máquinas
    ids = [f"MAQ-{i+1:03d}" for i in range(len(vibracao))]
    
    df = pd.DataFrame({
        'ID_Maquina': ids,
        'Frequencia_Vibracao': vibracao,
        'Pico_Temperatura': temperatura
    })
    
    return df

df = gerar_dados()

# -----------------------------------------------------------------------------
# 2. BARRA LATERAL (CONTROLES E HYPERPARÂMETROS)
# -----------------------------------------------------------------------------
st.sidebar.header("Parâmetros do DBSCAN")
st.sidebar.markdown("Ajuste os valores para recalibrar a sensibilidade da detecção de anomalias.")

# Controles para ajustar o DBSCAN
eps = st.sidebar.slider(
    "eps (Epsilon - Raio de vizinhança):",
    min_value=0.1,
    max_value=2.0,
    value=0.4,
    step=0.05,
    help="Distância máxima entre dois pontos para que sejam considerados vizinhos."
)

min_samples = st.sidebar.slider(
    "min_samples (Mínimo de amostras):",
    min_value=2,
    max_value=20,
    value=5,
    step=1,
    help="Número mínimo de pontos no raio 'eps' para formar uma região densa (cluster)."
)

# -----------------------------------------------------------------------------
# 3. PROCESSAMENTO E ALGORITMO DBSCAN
# -----------------------------------------------------------------------------
X = df[['Frequencia_Vibracao', 'Pico_Temperatura']].values

# Normalização dos dados (fundamental para algoritmos baseados em distância como DBSCAN)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicação do DBSCAN
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
labels = dbscan.fit_predict(X_scaled)

# Adiciona o resultado dos clusters ao dataframe
df['Cluster'] = labels
# No DBSCAN, a label -1 indica ruído/outlier
df['Status'] = df['Cluster'].apply(lambda x: 'Anomalia (Outlier)' if x == -1 else f'Cluster {x}')

# -----------------------------------------------------------------------------
# 4. EXIBIÇÃO DE MÉTRICAS RÁPIDAS
# -----------------------------------------------------------------------------
col1, col2, col3 = st.columns(3)
total_maquinas = len(df)
total_anomalias = (df['Cluster'] == -1).sum()
percentual_anomalias = (total_anomalias / total_maquinas) * 100

col1.metric("Total de Máquinas Monitoradas", total_maquinas)
col2.metric("Anomalias Detectadas", total_anomalias, delta_color="inverse")
col3.metric("Taxa de Anomalias", f"{percentual_anomalias:.1f}%")

st.markdown("---")

# -----------------------------------------------------------------------------
# 5. GRÁFICO DE DISPERSÃO COM SEABORN
# -----------------------------------------------------------------------------
st.subheader("📊 Mapeamento de Padrões e Anomalias")

# Criação da paleta de cores destacando outliers (-1) em preto
unique_clusters = sorted(df['Cluster'].unique())
palette = {}

# Paleta base do Seaborn para os clusters normais
colors = sns.color_palette("tab10", n_colors=len(unique_clusters))

for i, cluster in enumerate(unique_clusters):
    if cluster == -1:
        palette[cluster] = "#000000"  # Preto absoluto para outliers
    else:
        palette[cluster] = colors[i]

# Criação da figura Matplotlib / Seaborn
fig, ax = plt.subplots(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x='Frequencia_Vibracao',
    y='Pico_Temperatura',
    hue='Cluster',
    palette=palette,
    style=(df['Cluster'] == -1),  # Muda o formato do marcador para outliers
    markers={True: 'X', False: 'o'},
    s=80,
    ax=ax
)

# Estilização do gráfico
ax.set_title("Frequência de Vibração vs. Pico de Temperatura", fontsize=14, pad=15)
ax.set_xlabel("Frequência de Vibração (Hz)", fontsize=11)
ax.set_ylabel("Pico de Temperatura (°C)", fontsize=11)
ax.grid(True, linestyle="--", alpha=0.5)

# Ajuste da legenda
handles, labels_legend = ax.get_legend_handles_labels()
new_labels = ["Anomalia (Outlier)" if l == "-1" else f"Cluster {l}" for l in labels_legend]
ax.legend(handles=handles, labels=new_labels, title="Classificação", bbox_to_anchor=(1.02, 1), loc='upper left')

st.pyplot(fig)

# -----------------------------------------------------------------------------
# 6. TABELA INTERATIVA DE MÁQUINAS ANÔMALAS
# -----------------------------------------------------------------------------
st.subheader("🚨 Máquinas com Comportamento Anômalo")

df_anomalias = df[df['Cluster'] == -1][['ID_Maquina', 'Frequencia_Vibracao', 'Pico_Temperatura']]

if not df_anomalias.empty:
    st.warning(f"Atenção: Foram identificadas {len(df_anomalias)} máquinas operando fora dos padrões tolerados.")
    
    # Exibe a tabela interativa do Streamlit
    st.dataframe(
        df_anomalias.style.format({
            'Frequencia_Vibracao': '{:.2f} Hz',
            'Pico_Temperatura': '{:.2f} °C'
        }),
        use_container_width=True
    )
else:
    st.success("Nenhuma anomalia detectada com os parâmetros atuais do DBSCAN.")
