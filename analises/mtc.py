import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Colunas utilizadas (em todas as analises)
colunas = [
    "pontos",
    "total_rebotes",
    "assistencias",
    "roubos_bola",
    "bloqueios",
    "faltas_pessoais",
    "cestas_3_pontos_convertidas"
]


# Filtra todos os dados da equipe pelo nome
def filtrarBasePorTime (df: pd.DataFrame, time: str):
    return df[df["nome_time"] == time]


# Retorna a média, moda e mediana por variável
def mediaModaMediana(df: pd.DataFrame, time: str):
    dados_time = filtrarBasePorTime(df, time)

    # Calculos em si
    estatisticas = {}

    for coluna in colunas:
        estatisticas[coluna] = {
            "media": round(dados_time[coluna].mean(), 2),
            "mediana": round(dados_time[coluna].median(), 2),
            "moda": dados_time[coluna].mode().tolist()
        }

    return estatisticas


# Retorna os quartis, iqr e outliers por variável
def quartisOutliers(df: pd.DataFrame, time: str):
    dados_time = filtrarBasePorTime(df, time)

    resultados = {}

    for coluna in colunas:
        # Obtenção dos quartis (Q1 e Q3)
        q1 = dados_time[coluna].quantile(0.25)
        q3 = dados_time[coluna].quantile(0.75)

        # Cálculo dos limites inferiores e superiores
        iqr = q3 - q1
        limite_inferior = q1 - 1.5 * iqr
        limite_superior = q3 + 1.5 * iqr

        # Seleção dos outliers
        outliers = dados_time[
            (dados_time[coluna] < limite_inferior) |
            (dados_time[coluna] > limite_superior)
        ]

        # Organização dos dados
        resultados[coluna] = {
            "q1": round(q1, 2),
            "q3": round(q3, 2),
            "iqr": round(iqr, 2),
            "limite_inferior": round(limite_inferior, 2),
            "limite_superior": round(limite_superior, 2),
            "quantidade": len(outliers),
            "outliers": outliers,
        }

    return resultados


# Retorna os outliers separadamente
def listaOutliers(quartisOutliers: dict, metadados):
    lista_outliers = []

    for variavel, resultado in quartisOutliers.items():

        outliers = resultado["outliers"]

        for _, linha in outliers.iterrows():
            lista_outliers.append({
                "variavel": metadados[variavel]["nome"],
                "data": linha["data_partida"],
                "confronto": linha["confronto"],
                "valor": linha[variavel]
            })

    return lista_outliers


# Retorna algumas medidas de dispersão por variável
def medidasDispersao(df: pd.DataFrame, time: str):
    dados_time = filtrarBasePorTime(df, time)

    resultados = {}

    for coluna in colunas:
        menor = dados_time[coluna].min()
        maior = dados_time[coluna].max()

        resultados[coluna] = {
            "menor_valor": round(menor, 2),
            "maior_valor": round(maior, 2),
            "amplitude": round(maior - menor, 2),
            "desvio_padrao": round(dados_time[coluna].std(), 2),
            "variancia": round(dados_time[coluna].var(), 2)
        }

    return resultados


# Retorna um gráfico Boxplot para cada variável
def boxplotMtc(df: pd.DataFrame, time: str):
    dados_time = filtrarBasePorTime(df, time)
    resultados = quartisOutliers(df, time)

    fig = go.Figure()

    for coluna in colunas:
        resultado = resultados[coluna]

        q1 = resultado["q1"]
        q3 = resultado["q3"]
        limite_inferior = resultado["limite_inferior"]
        limite_superior = resultado["limite_superior"]

        valores = dados_time[coluna].dropna()

        if valores.empty:
            continue

        # Identificação dos outliers
        outliers = valores[
            (valores < limite_inferior) |
            (valores > limite_superior)
        ]

        # Valores que não são outliers
        valores_validos = valores[
            (valores >= limite_inferior) &
            (valores <= limite_superior)
        ]

        if valores_validos.empty:
            continue

        media = valores.mean()
        moda = valores.mode().tolist()

        # Boxplot
        fig.add_trace(go.Box(
            name=coluna,
            x=[coluna],
            q1=[q1],
            median=[valores.median()],
            q3=[q3],
            lowerfence=[valores_validos.min()],
            upperfence=[valores_validos.max()],
            boxpoints=False,
            showlegend=False
        ))

        # Média
        fig.add_trace(go.Scatter(
            x=[coluna],
            y=[media],
            mode="markers",
            name="Média",
            marker=dict(
                symbol="diamond",
                size=10,
                color="red"
            ),
            showlegend=(coluna == colunas[0]),
            hovertemplate="Média: %{y:.2f}<extra></extra>"
        ))

        # Moda
        if moda:
            fig.add_trace(go.Scatter(
                x=[coluna] * len(moda),
                y=moda,
                mode="markers",
                name="Moda",
                marker=dict(
                    symbol="triangle-up",
                    size=10,
                    color="green"
                ),
                showlegend=(coluna == colunas[0]),
                hovertemplate="Moda: %{y:.2f}<extra></extra>"
            ))

        # Outliers
        if not outliers.empty:
            fig.add_trace(go.Scatter(
                x=[coluna] * len(outliers),
                y=outliers,
                mode="markers",
                name="Outliers",
                marker=dict(
                    symbol="circle",
                    size=8,
                    color="black"
                ),
                showlegend=(coluna == colunas[0]),
                hovertemplate="Outlier: %{y}<extra></extra>"
            ))

    fig.update_layout(
        title=f"Distribuição das estatísticas - {time}",
        xaxis_title="Variáveis",
        yaxis_title="Valores",
        xaxis=dict(
            type="category",
            categoryorder="array",
            categoryarray=colunas
        ),
        legend_title="Medidas estatísticas"
    )

    return fig

# Retorna um grafico comparativo dos dois times para uma variavel selecionada
def graficoComparacao(df: pd.DataFrame, variavel: str):
    resultados = []

    for time in ["Cleveland Cavaliers", "Brooklyn Nets"]:

        dados = df[df["nome_time"] == time]

        # Medidas de tendência central
        central = mediaModaMediana(df, time)

        # Medidas de dispersão
        dispersao = medidasDispersao(df, time)

        # Quartis e outliers
        quartis = quartisOutliers(df, time)

        # Recupera os resultados da variável selecionada
        dados_central = central[variavel]
        dados_dispersao = dispersao[variavel]
        dados_quartis = quartis[variavel]

        resultados.append({
            "Time": time,
            "Média": dados_central["media"],
            "Mediana": dados_central["mediana"],
            "Moda": dados_central["moda"],
            "Menor valor": dados_dispersao["menor_valor"],
            "Maior valor": dados_dispersao["maior_valor"],
            "Amplitude": dados_dispersao["amplitude"],
            "Desvio padrão": dados_dispersao["desvio_padrao"],
            "Variância": dados_dispersao["variancia"],
            "Q1": dados_quartis["q1"],
            "Q3": dados_quartis["q3"],
            "IQR": dados_quartis["iqr"],
            "Limite inferior": dados_quartis["limite_inferior"],
            "Limite superior": dados_quartis["limite_superior"],
            "Quantidade de outliers": dados_quartis["quantidade"]
        })

    df_resultados = pd.DataFrame(resultados)

    df_grafico = df_resultados.melt(
        id_vars="Time",
        var_name="Medida estatística",
        value_name="Valor"
    )

    fig = px.bar(
        df_grafico,
        x="Medida estatística",
        y="Valor",
        color="Time",
        barmode="group",
        title=f"Comparação entre equipes — {variavel}",
        text_auto=".2f"
    )

    fig.update_layout(
        xaxis_title="Medidas estatísticas",
        yaxis_title="Valor",
        legend_title="Equipe",
        template="plotly_white",
        xaxis_tickangle=-45,
        height=600
    )

    return fig