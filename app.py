import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Configuração da página do Streamlit
st.set_page_config(
    page_title="Dashboard Preditivo 3D",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ Dashboard de Manutenção Preditiva - Análise 3D")
st.markdown("""
Este painel utiliza dados simulados de sensores de chão de fábrica (**Frequência de Vibração**, **Pico de Temperatura** e **Nível de Ruído/Pressão**) 
para identificar comportamentos normais e detectar falhas em um espaço tridimensional usando **DBSCAN**.
""")

# -----------------------------------------------------------------------------
# 1. GERAÇÃO DOS DADOS (NumPy / Pandas)
# -----------------------------------------------------------------------------
@st.cache_data
def gerar_dados(n_samples=300, seed=42):
    np.random.seed(seed)
    
    # Cluster 1: Funcionamento Normal - Operação Leve
    vib_1 = np.random.normal(loc=30, scale=4, size=int(n_samples * 0.45))
    temp_1 = np.random.normal(loc=60, scale=3, size=int(n_samples * 0.45))
    ruido_1 = np.random.normal(loc=40, scale=2, size=int(n_samples * 0.45))
    
    # Cluster 2: Funcionamento Normal - Carga Alta
    vib_2 = np.random.normal(loc=65, scale=5, size=int(n_samples * 0.45))
    temp_2 = np.random.normal(loc=85, scale=4, size=int(n_samples * 0.45))
    ruido_2 = np.random.normal(loc=70, scale=3, size=int(n_samples * 0.45))
    
    # Outliers (Anomalias Intencionais)
    n_outliers = int(n_samples * 0.10)
    vib_out = np.random.uniform(10, 100, size=n_outliers)
    temp_out = np.random.uniform(40, 120, size=n_outliers)
    ruido_out = np.random.uniform(20, 100, size=n_outliers)
    
    # Concatenação
    vibracao = np.concatenate([vib_1, vib_2, vib_out])
    temperatura = np.concatenate([temp_1, temp_2, temp_out])
    ruido = np.concatenate([ruido_1, ruido_2, ruido_out])
    
    ids = [f"MAQ-{i+1:03d}" for i in range(len(vibracao))]
    
    df = pd.DataFrame({
        'ID_Maquina': ids,
        'Frequencia_Vibracao': vibracao,
        'Pico_Temperatura': temperatura,
        'Nivel_Ruido': ruido
    })
    
    return df

df = gerar_dados()

# -----------------------------------------------------------------------------
# 2. CONTROLES NA BARRA LATERAL
# -----------------------------------------------------------------------------
st.sidebar.header("🎛️ Parâmetros do DBSCAN")

eps = st.sidebar.slider(
    "eps (Epsilon):",
    min_value=0.1,
    max_value=2.0,
    value=0.5,
    step=0.05,
    help="Raio máximo de distância entre pontos para formar um agrupamento."
)

min_samples = st.sidebar.slider(
    "min_samples:",
    min_value=2,
    max_value=20,
    value=5,
    step=1,
    help="Quantidade mínima de amostras dentro do raio 'eps'."
)

# -----------------------------------------------------------------------------
# 3. ALGORITMO DBSCAN E ENGENHARIA DE RECURSOS
# -----------------------------------------------------------------------------
X = df[['Frequencia_Vibracao', 'Pico_Temperatura', 'Nivel_Ruido']].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

dbscan = DBSCAN(eps=eps, min_samples=min_samples)
df['Cluster'] = dbscan.fit_predict(X_scaled)

# Mapeamento do status e rótulos
df['Status'] = df['Cluster'].apply(lambda x: 'Anomalia (Outlier)' if x == -1 else f'Cluster Normal {x}')

# -----------------------------------------------------------------------------
# 4. PAINEL DE MÉTRICAS
# -----------------------------------------------------------------------------
col1, col2, col3 = st.columns(3)
total = len(df)
anomalias = (df['Cluster'] == -1).sum()
taxa = (anomalias / total) * 100

col1.metric("Máquinas Monitoradas", total)
col2.metric("Anomalias Encontradas", anomalias)
col3.metric("Taxa de Anomalias", f"{taxa:.1f}%")

st.markdown("---")

# -----------------------------------------------------------------------------
# 5. GRÁFICO 3D COM PLOTLY (Sem Legenda e com Marcadores com Informações)
# -----------------------------------------------------------------------------
st.subheader("🧊 Gráfico 3D de Vibração x Temperatura x Ruído")

# Definição do mapa de cores vibrantes (Anomalias em Preto, Clusters em cores destacadas)
unique_status = df['Status'].unique()
vibrant_colors = ['#FF5722', '#00E676', '#29B6F6', '#AB47BC', '#FFCA28', '#E91E63']

color_discrete_map = {}
color_idx = 0

for status in sorted(unique_status):
    if status == 'Anomalia (Outlier)':
        color_discrete_map[status] = '#000000'  # Anomalias sempre em PRETO
    else:
        color_discrete_map[status] = vibrant_colors[color_idx % len(vibrant_colors)]
        color_idx += 1

# Criação da figura 3D
fig = px.scatter_3d(
    df,
    x='Frequencia_Vibracao',
    y='Pico_Temperatura',
    z='Nivel_Ruido',
    color='Status',
    color_discrete_map=color_discrete_map,
    hover_name='ID_Maquina',
    hover_data={
        'Frequencia_Vibracao': ':.2f',
        'Pico_Temperatura': ':.2f',
        'Nivel_Ruido': ':.2f',
        'Status': True
    },
    opacity=0.85
)

# Atualização de layout: Oculta a legenda conforme solicitado e ajusta visualização
fig.update_layout(
    showlegend=False,  # Sem legenda
    margin=dict(l=0, r=0, b=0, t=30),
    scene=dict(
        xaxis_title='Vibração (Hz)',
        yaxis_title='Temperatura (°C)',
        zaxis_title='Ruído (dB)',
        xaxis=dict(backgroundcolor="rgba(240, 240, 240, 0.5)"),
        yaxis=dict(backgroundcolor="rgba(240, 240, 240, 0.5)"),
        zaxis=dict(backgroundcolor="rgba(240, 240, 240, 0.5)")
    ),
    height=600
)

# Ajuste do tamanho dos pontos no gráfico
fig.update_traces(marker=dict(size=6))

st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# 6. TABELA INTERATIVA DE FILTRAGEM (Normais vs Anomalias)
# -----------------------------------------------------------------------------
st.subheader("📋 Tabela Interativa de Monitoramento")

# Filtro do usuário para a tabela
opcoes_filtro = st.multiselect(
    "Filtrar máquinas por Status:",
    options=list(df['Status'].unique()),
    default=list(df['Status'].unique()),
    help="Selecione um ou mais status para filtrar os dados da tabela abaixo."
)

# Aplicação do filtro
df_filtrado = df[df['Status'].isin(opcoes_filtro)]

if not df_filtrado.empty:
    # Exibe tabela interativa com ordenação e busca integrada
    st.dataframe(
        df_filtrado[['ID_Maquina', 'Status', 'Frequencia_Vibracao', 'Pico_Temperatura', 'Nivel_Ruido']].style.format({
            'Frequencia_Vibracao': '{:.2f} Hz',
            'Pico_Temperatura': '{:.2f} °C',
            'Nivel_Ruido': '{:.2f} dB'
        }),
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("Nenhuma máquina encontrada para os filtros selecionados.")
