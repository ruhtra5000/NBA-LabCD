import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from analises.tratamento import df

# Colunas analisadas no projeto e seus metadados (nome legível e unidade)
colunas = [
    "pontos",
    "total_rebotes",
    "assistencias",
    "roubos_bola",
    "bloqueios",
    "faltas_pessoais",
    "cestas_3_pontos_convertidas"
]

METADADOS_VARIAVEIS = {
    "pontos": {
        "nome": "Pontos (PTS)",
        "unidade": "pts/partida"
    },
    "total_rebotes": {
        "nome": "Total de Rebotes (REB)",
        "unidade": "reb/partida"
    },
    "assistencias": {
        "nome": "Assistências (AST)",
        "unidade": "ast/partida"
    },
    "roubos_bola": {
        "nome": "Roubos de Bola (STL)",
        "unidade": "stl/partida"
    },
    "bloqueios": {
        "nome": "Bloqueios (BLK)",
        "unidade": "blk/partida"
    },
    "faltas_pessoais": {
        "nome": "Faltas Pessoais (PF)",
        "unidade": "faltas/partida"
    },
    "cestas_3_pontos_convertidas": {
        "nome": "Cestas de 3 Pontos (FG3M)",
        "unidade": "cestas/partida"
    }
}

# Carregar e padronizar dados
dados = df[colunas].copy()
dados = dados.dropna()

scaler = StandardScaler()
dados_padronizados = scaler.fit_transform(dados)

# Executa algoritmo K-Means
def executarKMeans(k: int):
    # Criação do modelo
    modelo = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=10,
        random_state=42
    )

    # Executar a clusterização
    clusters = modelo.fit_predict(dados_padronizados)

    # Associar os clusters às partidas originais
    resultado = df.loc[dados.index].copy()
    resultado["cluster"] = clusters

    return resultado, modelo

# Categoriza os clusters nas categorias solicitadas
def categorizarClustersKMeans(resultado, modelo):
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

    distancias = np.linalg.norm(modelo.cluster_centers_, axis=1)

    categorias.append({
        "nome": "Características equilibradas",
        "descricao": "Partidas com comportamento intermediário ou relativamente equilibrado nas variáveis analisadas.",
        "cluster": int(distancias.argmin())
    })

    return categorias

# Gera o grafico de dispersão em relação a 2 das 7 variáveis analisadas
def graficoDispersao(resultado, modelo, eixo_x, eixo_y):
    dados = resultado.copy()
    dados["cluster"] = dados["cluster"].astype(str)

    nome_x = METADADOS_VARIAVEIS[eixo_x]["nome"]
    nome_y = METADADOS_VARIAVEIS[eixo_y]["nome"]

    unidade_x = METADADOS_VARIAVEIS[eixo_x]["unidade"]
    unidade_y = METADADOS_VARIAVEIS[eixo_y]["unidade"]

    # Gráfico das partidas
    fig = px.scatter(
        dados,
        x=eixo_x,
        y=eixo_y,
        color="cluster",
        title="Distribuição das partidas por cluster",
        labels={
            eixo_x: f"{nome_x} ({unidade_x})",
            eixo_y: f"{nome_y} ({unidade_y})",
            "cluster": "Cluster"
        },
        color_discrete_sequence=px.colors.qualitative.Set2
    )

    # Converter os centroides para a escala original
    centroides = pd.DataFrame(
        scaler.inverse_transform(modelo.cluster_centers_),
        columns=colunas
    )

    # Adicionar os centroides ao gráfico
    fig.add_scatter(
        x=centroides[eixo_x],
        y=centroides[eixo_y],
        mode="markers",
        name="Centroides",
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

# Gera o bloxplot por cluster baseado numa variável por vez
def graficoBoxplotCluster(resultado, variavel):
    dados = resultado.copy()
    dados["cluster"] = dados["cluster"].astype(str)

    nome_variavel = METADADOS_VARIAVEIS[variavel]["nome"]
    unidade = METADADOS_VARIAVEIS[variavel]["unidade"]

    fig = px.box(
        dados,
        x="cluster",
        y=variavel,
        color="cluster",
        title=f"Distribuição de {nome_variavel} por cluster",
        labels={
            "cluster": "Cluster",
            variavel: f"{nome_variavel} ({unidade})"
        },
        color_discrete_sequence=px.colors.qualitative.Set2,
        points="outliers"
    )

    fig.update_layout(
        template="plotly_white",
        showlegend=False
    )

    return fig

# Gera um radarchart com os dados de todos os clusters
def graficoRadarClusters(resultado):
    # Selecionar as variáveis utilizadas no K-means
    dados = resultado[colunas].copy()

    # Padronizar os dados
    dados_padronizados = pd.DataFrame(
        scaler.transform(dados),
        columns=colunas,
        index=dados.index
    )

    # Associar os clusters às observações
    dados_padronizados["cluster"] = resultado["cluster"]

    # Calcular a média de cada variável por cluster
    medias = dados_padronizados.groupby("cluster")[colunas].mean()

    fig = go.Figure()

    # Criar uma linha para cada cluster
    for cluster in medias.index:

        valores = medias.loc[cluster].tolist()

        # Repetir o primeiro valor para fechar o radar
        valores.append(valores[0])

        fig.add_trace(go.Scatterpolar(
            r=valores,
            theta=colunas + [colunas[0]],
            fill="toself",
            name=f"Cluster {cluster}"
        ))

    # Utilizar os nomes das variáveis nos eixos
    nomes = [
        METADADOS_VARIAVEIS[coluna]["nome"]
        for coluna in colunas
    ]

    fig.update_layout(
        title="Perfil dos clusters por variável",
        template="plotly_white",
        polar=dict(
            radialaxis=dict(
                visible=True,
                title="Média padronizada"
            ),
            angularaxis=dict(
                tickmode="array",
                tickvals=colunas,
                ticktext=nomes
            )
        ),
        legend_title="Clusters"
    )

    return fig

# Gera um histograma para os clusters (c/ sobreposição) baseado numa variavel por vez
def graficoHistogramaClusters(resultado, variavel):
    dados = resultado.copy()
    dados["cluster"] = dados["cluster"].astype(str)

    nome_variavel = METADADOS_VARIAVEIS[variavel]["nome"]
    unidade = METADADOS_VARIAVEIS[variavel]["unidade"]

    fig = px.histogram(
        dados,
        x=variavel,
        color="cluster",
        nbins=20,
        barmode="overlay",
        opacity=0.65,
        title=f"Distribuição de {nome_variavel} por cluster",
        labels={
            variavel: f"{nome_variavel} ({unidade})",
            "count": "Quantidade de partidas",
            "cluster": "Cluster"
        },
        color_discrete_sequence=px.colors.qualitative.Set2
    )

    fig.update_layout(
        template="plotly_white",
        legend_title="Clusters",
        xaxis_title=f"{nome_variavel} ({unidade})",
        yaxis_title="Quantidade de partidas"
    )

    return fig

# Gera um grafico de linhas indicando a evolução de cada cluster durante a temporada
def graficoEvolucaoClusters(resultado, periodo="W"):
    dados = resultado.copy()
    dados["cluster"] = dados["cluster"].astype(str)

    # Agrupar as partidas por período e cluster
    evolucao = (
        dados.groupby([
            pd.Grouper(key="data_partida", freq=periodo),
            "cluster"
        ])
        .size()
        .reset_index(name="quantidade")
    )

    fig = px.line(
        evolucao,
        x="data_partida",
        y="quantidade",
        color="cluster",
        markers=True,
        title="Evolução dos clusters durante a temporada",
        labels={
            "data_partida": "Data",
            "quantidade": "Quantidade de observações",
            "cluster": "Cluster"
        },
        color_discrete_sequence=px.colors.qualitative.Set2
    )

    fig.update_layout(
        template="plotly_white",
        legend_title="Clusters",
        xaxis_title="Data da partida",
        yaxis_title="Quantidade de observações",
        hovermode="x unified"
    )

    return fig

# Gera um grafico que indica a quantidade de partidas por cluster
def graficoQuantidadeClusters(resultado):
    # Contabilizar observações por cluster
    quantidades = (
        resultado["cluster"]
        .value_counts()
        .sort_index()
        .reset_index()
    )

    quantidades.columns = ["cluster", "quantidade"]

    quantidades["cluster"] = quantidades["cluster"].astype(str)

    fig = px.bar(
        quantidades,
        x="cluster",
        y="quantidade",
        color="cluster",
        text="quantidade",
        title="Quantidade de observações por cluster",
        labels={
            "cluster": "Cluster",
            "quantidade": "Quantidade de observações"
        },
        color_discrete_sequence=px.colors.qualitative.Set2
    )

    fig.update_traces(
        textposition="outside"
    )

    fig.update_layout(
        template="plotly_white",
        showlegend=False,
        xaxis_title="Cluster",
        yaxis_title="Quantidade de observações"
    )

    return fig

# Gera um grafico comparativo da função K-Means alterando os valores de K
def graficoComparacaoK():
    resultados = []

    # Executar K-means para cada valor de K
    for k in [2, 3, 4]:

        resultado, modelo = executarKMeans(k)

        # Contabilizar observações por cluster
        quantidades = resultado["cluster"].value_counts().sort_index()

        for cluster, quantidade in quantidades.items():

            resultados.append({
                "K": f"K = {k}",
                "cluster": f"Cluster {cluster}",
                "quantidade": quantidade
            })

    dados = pd.DataFrame(resultados)

    fig = px.bar(
        dados,
        x="K",
        y="quantidade",
        color="cluster",
        barmode="group",
        text="quantidade",
        title="Comparação da quantidade de observações por cluster",
        labels={
            "K": "Número de clusters",
            "quantidade": "Quantidade de observações",
            "cluster": "Cluster"
        },
        color_discrete_sequence=px.colors.qualitative.Set2
    )

    fig.update_traces(
        textposition="outside"
    )

    fig.update_layout(
        template="plotly_white",
        legend_title="Clusters"
    )

    return fig

# Calcula Silhouette Score e gera um grafico comparativo
def graficoComparacaoSilhouette():
    resultados = []

    for k in [2, 3, 4]:

        resultado, modelo = executarKMeans(k)

        score = silhouette_score(
            dados_padronizados,
            modelo.labels_
        )

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
        title="Comparação do Silhouette Score por K",
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
