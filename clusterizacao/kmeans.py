import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
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
def categorizarClustersKMeans(resultado: pd.DataFrame, modelo: KMeans):
    # Calcular as médias de cada variável por cluster
    medias = resultado.groupby("cluster")[colunas].mean()

    # Identificar o cluster com maior média em cada variável
    categorias = {
        "Maior pontuação":                    medias["pontos"].idxmax(),
        "Maior número de assistências":       medias["assistencias"].idxmax(),
        "Maior número de rebotes":            medias["total_rebotes"].idxmax(),
        "Maior número de cestas de 3 pontos": medias["cestas_3_pontos_convertidas"].idxmax(),
        "Maior número de faltas":             medias["faltas_pessoais"].idxmax()
    }

    # Identificar o cluster mais próximo do perfil médio geral
    distancias = (modelo.cluster_centers_ ** 2).sum(axis=1) ** 0.5

    categorias["Características equilibradas"] = distancias.argmin()

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