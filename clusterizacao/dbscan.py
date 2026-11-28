import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from analises.tratamento import df
from clusterizacao.kmeans import (
    colunas,
    METADADOS_VARIAVEIS,
    dados,
    scaler,
    dados_padronizados,
    graficoBoxplotCluster,
    graficoRadarClusters,
    graficoHistogramaClusters,
    graficoEvolucaoClusters,
    graficoQuantidadeClusters,
)

# Configurações pré-definidas de DBSCAN mapeadas para K = 2, 3 e 4
EPS_PADRAO: dict[int, float] = {2: 1.15, 3: 1.35, 4: 1.60}
MIN_SAMPLES_PADRAO: dict[int, int] = {2: 4, 3: 5, 4: 6}

CONFIGURACOES_DBSCAN = {
    2: {"eps": 1.15, "min_samples": 4, "rotulo": "Config 1 (eps = 1.15, min_samples = 4)"},
    3: {"eps": 1.35, "min_samples": 5, "rotulo": "Config 2 (eps = 1.35, min_samples = 5)"},
    4: {"eps": 1.60, "min_samples": 6, "rotulo": "Config 3 (eps = 1.60, min_samples = 6)"},
}


def _calcularCentroidesPadronizadosDBSCAN(clusters: pd.Series) -> pd.DataFrame:
    df_temp = pd.DataFrame(dados_padronizados, columns=colunas, index=dados.index)
    df_temp["cluster"] = clusters
    sem_ruido = df_temp[df_temp["cluster"] != -1]
    if sem_ruido.empty:
        sem_ruido = df_temp
    return sem_ruido.groupby("cluster")[colunas].mean().sort_index()


# Executa algoritmo DBSCAN
def executarDBSCAN(k: int = 3, eps: float | None = None, min_samples: int | None = None):
    eps_usado = eps if eps is not None else EPS_PADRAO.get(k, 1.35)
    min_samples_usado = min_samples if min_samples is not None else MIN_SAMPLES_PADRAO.get(k, 5)

    modelo = DBSCAN(
        eps=eps_usado,
        min_samples=min_samples_usado,
        metric="euclidean"
    )

    clusters = modelo.fit_predict(dados_padronizados)

    resultado = df.loc[dados.index].copy()
    resultado["cluster"] = clusters

    return resultado, modelo


# Categoriza os clusters nas categorias solicitadas
def categorizarClustersDBSCAN(resultado: pd.DataFrame, modelo: DBSCAN):
    dados_validos = resultado[resultado["cluster"] != -1]
    if dados_validos.empty:
        dados_validos = resultado

    medias = dados_validos.groupby("cluster")[colunas].mean()

    categorias = [
        {
            "nome": "Maior pontuação",
            "descricao": "Partidas nas quais a equipe apresentou maior produção ofensiva.",
            "cluster": medias["pontos"].idxmax()
        },
        {
            "nome": "Maior número de assistências",
            "descricao": "Partidas caracterizadas por maior distribuição de bola e participação coletiva.",
            "cluster": medias["assistencias"].idxmax()
        },
        {
            "nome": "Maior número de rebotes",
            "descricao": "Partidas em que a equipe apresentou maior desempenho nos rebotes.",
            "cluster": medias["total_rebotes"].idxmax()
        },
        {
            "nome": "Maior número de cestas de três pontos",
            "descricao": "Partidas com maior produção de arremessos de longa distância.",
            "cluster": medias["cestas_3_pontos_convertidas"].idxmax()
        },
        {
            "nome": "Maior número de faltas",
            "descricao": "Partidas em que a equipe apresentou maior quantidade de faltas pessoais.",
            "cluster": medias["faltas_pessoais"].idxmax()
        }
    ]

    centroides_df = _calcularCentroidesPadronizadosDBSCAN(resultado["cluster"])
    distancias = np.linalg.norm(centroides_df.to_numpy(), axis=1)
    idx_equilibrado = int(distancias.argmin())
    cluster_equilibrado = centroides_df.index[idx_equilibrado]

    categorias.append({
        "nome": "Características equilibradas",
        "descricao": "Partidas com comportamento intermediário ou relativamente equilibrado nas variáveis analisadas.",
        "cluster": cluster_equilibrado
    })

    return categorias


# Gera o gráfico de dispersão em relação a 2 das 7 variáveis analisadas
def graficoDispersaoDBSCAN(resultado: pd.DataFrame, modelo: DBSCAN, eixo_x: str, eixo_y: str):
    dados_plot = resultado.copy()
    dados_plot["cluster"] = dados_plot["cluster"].apply(
        lambda c: "Ruído (-1)" if c == -1 else str(c)
    )

    nome_x = METADADOS_VARIAVEIS[eixo_x]["nome"]
    nome_y = METADADOS_VARIAVEIS[eixo_y]["nome"]

    unidade_x = METADADOS_VARIAVEIS[eixo_x]["unidade"]
    unidade_y = METADADOS_VARIAVEIS[eixo_y]["unidade"]

    fig = px.scatter(
        dados_plot,
        x=eixo_x,
        y=eixo_y,
        color="cluster",
        title="Distribuição das partidas por cluster (DBSCAN)",
        labels={
            eixo_x: f"{nome_x} ({unidade_x})",
            eixo_y: f"{nome_y} ({unidade_y})",
            "cluster": "Cluster"
        },
        color_discrete_sequence=px.colors.qualitative.Set2
    )

    centroides_df = _calcularCentroidesPadronizadosDBSCAN(resultado["cluster"])
    if not centroides_df.empty:
        centroides = pd.DataFrame(
            scaler.inverse_transform(centroides_df.to_numpy()),
            columns=colunas
        )

        fig.add_scatter(
            x=centroides[eixo_x],
            y=centroides[eixo_y],
            mode="markers",
            name="Centroides Médios",
            marker=dict(
                symbol="x",
                size=15,
                color="black",
                line=dict(width=3)
            )
        )

    fig.update_layout(
        template="plotly_white",
        legend_title="Clusters"
    )

    return fig


# Gera um gráfico comparativo do DBSCAN alterando as configurações
def graficoComparacaoKDBSCAN():
    resultados = []

    for k in [2, 3, 4]:
        eps_cfg = EPS_PADRAO[k]
        min_cfg = MIN_SAMPLES_PADRAO[k]
        resultado, _ = executarDBSCAN(k)
        quantidades = resultado["cluster"].value_counts().sort_index()

        for cluster, quantidade in quantidades.items():
            nome_cluster = "Ruído (-1)" if cluster == -1 else f"Cluster {cluster}"
            resultados.append({
                "K": f"eps={eps_cfg} (min={min_cfg})",
                "cluster": nome_cluster,
                "quantidade": quantidade
            })

    dados_comp = pd.DataFrame(resultados)

    fig = px.bar(
        dados_comp,
        x="K",
        y="quantidade",
        color="cluster",
        barmode="group",
        text="quantidade",
        title="Comparação da quantidade de observações por configuração (DBSCAN)",
        labels={
            "K": "Configuração (eps e min_samples)",
            "quantidade": "Quantidade de observações",
            "cluster": "Cluster"
        },
        color_discrete_sequence=px.colors.qualitative.Set2
    )

    fig.update_traces(textposition="outside")
    fig.update_layout(
        template="plotly_white",
        legend_title="Clusters"
    )

    return fig


# Calcula Silhouette Score e gera um gráfico comparativo para as configurações do DBSCAN
def graficoComparacaoSilhouetteDBSCAN():
    resultados = []

    for k in [2, 3, 4]:
        eps_cfg = EPS_PADRAO[k]
        min_cfg = MIN_SAMPLES_PADRAO[k]
        _, modelo = executarDBSCAN(k)
        rotulos = modelo.labels_
        n_labels = len(set(rotulos))

        if 2 <= n_labels < len(dados_padronizados):
            score = silhouette_score(dados_padronizados, rotulos)
        else:
            score = 0.0

        resultados.append({
            "K": f"eps={eps_cfg} (min={min_cfg})",
            "silhouette": score
        })

    dados_score = pd.DataFrame(resultados)

    fig = px.bar(
        dados_score,
        x="K",
        y="silhouette",
        text="silhouette",
        title="Comparação do Silhouette Score por Configuração (DBSCAN)",
        labels={
            "K": "Configuração (eps e min_samples)",
            "silhouette": "Silhouette Score"
        }
    )

    fig.update_traces(
        texttemplate="%{text:.3f}",
        textposition="outside"
    )

    fig.update_layout(
        template="plotly_white",
        yaxis_title="Silhouette Score"
    )

    return fig
