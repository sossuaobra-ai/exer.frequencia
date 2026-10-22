import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Manutenção Preditiva", page_icon="⚙️", layout="wide")

st.title("⚙️ Dashboard de Manutenção Preditiva")
st.caption("Monitoramento de sensores industriais com detecção de anomalias via DBSCAN")

@st.cache_data
def gerar_dados(seed: int = 42, n_maquinas: int = 250) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Leituras típicas de operação: dois grupos de comportamento normal.
    n1 = int(n_maquinas * 0.55)
    n2 = n_maquinas - n1
    vibracao_1 = rng.normal(32, 4.5, n1)
    temperatura_1 = rng.normal(62, 5, n1)
    vibracao_2 = rng.normal(48, 5, n2)
    temperatura_2 = rng.normal(78, 4.5, n2)

    df = pd.DataFrame({
        "Maquina": [f"M-{i:03d}" for i in range(1, n_maquinas + 1)],
        "Frequencia_Vibracao": np.r_[vibracao_1, vibracao_2],
        "Pico_Temperatura": np.r_[temperatura_1, temperatura_2],
    })

    # Outliers intencionais: vibração/temperatura muito elevadas ou combinações incomuns.
    indices = rng.choice(df.index, size=12, replace=False)
    df.loc[indices, "Frequencia_Vibracao"] = rng.uniform(68, 92, len(indices))
    df.loc[indices, "Pico_Temperatura"] = rng.uniform(95, 125, len(indices))
    return df

df = gerar_dados()

with st.sidebar:
    st.header("Parâmetros do DBSCAN")
    eps = st.slider("eps", min_value=0.1, max_value=2.5, value=0.35, step=0.05,
                    help="Raio de vizinhança aplicado aos dados padronizados.")
    min_samples = st.slider("min_samples", min_value=3, max_value=20, value=6, step=1,
                            help="Número mínimo de pontos para formar uma região densa.")
    st.divider()
    st.caption("Os dados são sintéticos. O DBSCAN é executado sobre as métricas padronizadas.")

features = ["Frequencia_Vibracao", "Pico_Temperatura"]
X = StandardScaler().fit_transform(df[features])
modelo = DBSCAN(eps=eps, min_samples=min_samples)
df["Cluster"] = modelo.fit_predict(X)
df["Status"] = np.where(df["Cluster"] == -1, "Anomalia", "Normal")
anomalias = df[df["Cluster"] == -1].copy()

col1, col2, col3 = st.columns(3)
col1.metric("Máquinas monitoradas", f"{len(df)}")
col2.metric("Anomalias detectadas", f"{len(anomalias)}")
col3.metric("Clusters encontrados", f"{df.loc[df['Cluster'] != -1, 'Cluster'].nunique()}")

st.subheader("Mapa de funcionamento das máquinas")
fig, ax = plt.subplots(figsize=(11, 6))
normais = df[df["Cluster"] != -1]
if not normais.empty:
    sns.scatterplot(
        data=normais,
        x="Frequencia_Vibracao",
        y="Pico_Temperatura",
        hue="Cluster",
        palette="tab10",
        s=75,
        alpha=0.85,
        ax=ax,
        legend="full",
    )
if not anomalias.empty:
    sns.scatterplot(
        data=anomalias,
        x="Frequencia_Vibracao",
        y="Pico_Temperatura",
        color="black",
        marker="X",
        s=120,
        label="Anomalia (ruído)",
        ax=ax,
    )
ax.set_xlabel("Frequência de vibração")
ax.set_ylabel("Pico de temperatura (°C)")
ax.set_title("Clusters DBSCAN — sensores de chão de fábrica")
ax.grid(True, alpha=0.2)
ax.legend(title="Grupo", bbox_to_anchor=(1.02, 1), loc="upper left")
fig.tight_layout()
st.pyplot(fig, use_container_width=True)
plt.close(fig)

st.subheader("Máquinas consideradas anômalas")
st.write("Pontos classificados pelo DBSCAN como ruído (rótulo de cluster `-1`).")
if anomalias.empty:
    st.success("Nenhuma anomalia foi identificada com os parâmetros atuais. Ajuste eps ou min_samples para explorar outros resultados.")
else:
    st.dataframe(
        anomalias[["Maquina", "Frequencia_Vibracao", "Pico_Temperatura", "Cluster", "Status"]]
        .sort_values(["Pico_Temperatura", "Frequencia_Vibracao"], ascending=False)
        .reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Frequencia_Vibracao": st.column_config.NumberColumn("Frequência de vibração", format="%.2f"),
            "Pico_Temperatura": st.column_config.NumberColumn("Pico de temperatura (°C)", format="%.2f"),
        },
    )
    st.download_button(
        "Baixar anomalias em CSV",
        data=anomalias[["Maquina", "Frequencia_Vibracao", "Pico_Temperatura", "Cluster", "Status"]].to_csv(index=False).encode("utf-8"),
        file_name="maquinas_anomalas.csv",
        mime="text/csv",
    )

with st.expander("Como interpretar"):
    st.markdown(
        "- **Cores:** representam clusters de operação identificados pelo DBSCAN.\n"
        "- **Preto (X):** pontos classificados como ruído/anomalias (`Cluster = -1`).\n"
        "- **eps:** aumenta ou reduz o raio de vizinhança; valores maiores tendem a agrupar mais pontos.\n"
        "- **min_samples:** define a densidade mínima necessária para formar um cluster.\n\n"
        "As métricas são padronizadas antes da clusterização para evitar que a escala de cada sensor domine o cálculo da distância."
    )
