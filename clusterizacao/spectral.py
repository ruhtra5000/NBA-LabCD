import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.cluster import SpectralClustering
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


def _calcularCentroidesPadronizadosSpectral(clusters: pd.Series) -> np.ndarray:
    df_temp = pd.DataFrame(dados_padronizados, columns=colunas, index=dados.index)
    df_temp["cluster"] = clusters
    medias = df_temp.groupby("cluster")[colunas].mean().sort_index()
    return medias.to_numpy()


# Executa algoritmo Spectral Clustering
def executarSpectral(k: int = 3):
    modelo = SpectralClustering(
        n_clusters=k,
        affinity="nearest_neighbors",
        n_neighbors=15,
        assign_labels="kmeans",
        random_state=42
    )

    clusters = modelo.fit_predict(dados_padronizados)

    resultado = df.loc[dados.index].copy()
    resultado["cluster"] = clusters

    return resultado, modelo


# Categoriza os clusters nas categorias solicitadas
def categorizarClustersSpectral(resultado: pd.DataFrame, modelo: SpectralClustering):
    medias = resultado.groupby("cluster")[colunas].mean()

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

    centroides_padronizados = _calcularCentroidesPadronizadosSpectral(resultado["cluster"])
    distancias = np.linalg.norm(centroides_padronizados, axis=1)

    categorias.append({
        "nome": "Características equilibradas",
        "descricao": "Partidas com comportamento intermediário ou relativamente equilibrado nas variáveis analisadas.",
        "cluster": int(distancias.argmin())
    })

    return categorias


# Gera o gráfico de dispersão em relação a 2 das 7 variáveis analisadas
def graficoDispersaoSpectral(resultado: pd.DataFrame, modelo: SpectralClustering, eixo_x: str, eixo_y: str):
    dados_plot = resultado.copy()
    dados_plot["cluster"] = dados_plot["cluster"].astype(str)

    nome_x = METADADOS_VARIAVEIS[eixo_x]["nome"]
    nome_y = METADADOS_VARIAVEIS[eixo_y]["nome"]

    unidade_x = METADADOS_VARIAVEIS[eixo_x]["unidade"]
    unidade_y = METADADOS_VARIAVEIS[eixo_y]["unidade"]

    fig = px.scatter(
        dados_plot,
        x=eixo_x,
        y=eixo_y,
        color="cluster",
        title="Distribuição das partidas por cluster (Spectral Clustering)",
        labels={
            eixo_x: f"{nome_x} ({unidade_x})",
            eixo_y: f"{nome_y} ({unidade_y})",
            "cluster": "Cluster"
        },
        color_discrete_sequence=px.colors.qualitative.Set2
    )

    centroides_padronizados = _calcularCentroidesPadronizadosSpectral(resultado["cluster"])
    centroides = pd.DataFrame(
        scaler.inverse_transform(centroides_padronizados),
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


# Gera um gráfico comparativo do Spectral Clustering alterando os valores de K
def graficoComparacaoKSpectral():
    resultados = []

    for k in [2, 3, 4]:
        resultado, _ = executarSpectral(k)
        quantidades = resultado["cluster"].value_counts().sort_index()

        for cluster, quantidade in quantidades.items():
            resultados.append({
                "K": f"K = {k}",
                "cluster": f"Cluster {cluster}",
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
        title="Comparação da quantidade de observações por cluster (Spectral Clustering)",
        labels={
            "K": "Número de clusters",
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


# Calcula Silhouette Score e gera um gráfico comparativo para K = 2, 3 e 4
def graficoComparacaoSilhouetteSpectral():
    resultados = []

    for k in [2, 3, 4]:
        _, modelo = executarSpectral(k)
        score = silhouette_score(dados_padronizados, modelo.labels_)

        resultados.append({
            "K": f"K = {k}",
            "silhouette": score
        })

    dados_score = pd.DataFrame(resultados)

    fig = px.bar(
        dados_score,
        x="K",
        y="silhouette",
        text="silhouette",
        title="Comparação do Silhouette Score por K (Spectral Clustering)",
        labels={
            "K": "Número de clusters",
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
