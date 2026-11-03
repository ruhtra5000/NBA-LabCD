import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from analises.assimetria import METADADOS_VARIAVEIS

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
    legendas_outliers = set()

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

        for nome, valor, cor, simbolo in [
            ("Q1", q1, "#4F46E5", "line-ew"),
            ("Mediana", valores.median(), "#0D9488", "line-ew"),
            ("Q3", q3, "#4F46E5", "line-ew"),
            ("Limite inferior", limite_inferior, "#7C3AED", "x"),
            ("Limite superior", limite_superior, "#7C3AED", "x"),
        ]:
            fig.add_trace(go.Scatter(
                x=[coluna], y=[valor], mode="markers", name=nome,
                marker=dict(symbol=simbolo, size=10, color=cor),
                showlegend=(coluna == colunas[0]),
                hovertemplate=nome+": %{y:.2f} "+METADADOS_VARIAVEIS[coluna]["unidade"]+"<extra></extra>"
            ))

        for nome, selecionados, simbolo in [
            ("Outliers inferiores", valores[valores < limite_inferior], "triangle-down"),
            ("Outliers superiores", valores[valores > limite_superior], "triangle-up"),
        ]:
            if selecionados.empty:
                continue
            fig.add_trace(go.Scatter(
                x=[coluna]*len(selecionados), y=selecionados, mode="markers", name=nome,
                marker=dict(symbol=simbolo, size=8, color="black"),
                showlegend=nome not in legendas_outliers,
                hovertemplate=nome+": %{y:.2f} "+METADADOS_VARIAVEIS[coluna]["unidade"]+"<extra></extra>"
            ))
            legendas_outliers.add(nome)

    fig.update_layout(
        title=f"Distribuição das estatísticas - {time}",
        xaxis_title="Variáveis",
        yaxis_title="Ocorrências por partida",
        xaxis=dict(
            type="category",
            categoryorder="array",
            categoryarray=colunas,
            tickmode="array", tickvals=colunas,
            ticktext=[METADADOS_VARIAVEIS[c]["nome"]+"<br>"+METADADOS_VARIAVEIS[c]["unidade"] for c in colunas]
        ),
        legend_title="Medidas estatísticas"
    )

    return fig

# Função para gerar um gráfico de comparação entre as duas equipes
def graficoComparacao(df: pd.DataFrame, variavel: str):
    resultados = []
    unidade = METADADOS_VARIAVEIS[variavel]["unidade"]
    unidades = [unidade, f"({unidade})²", "jogos"]
    for time in ["Cleveland Cavaliers", "Brooklyn Nets"]:
        central = mediaModaMediana(df, time)[variavel]
        dispersao = medidasDispersao(df, time)[variavel]
        quartis = quartisOutliers(df, time)[variavel]
        medidas = {
            "Média": central["media"], "Mediana": central["mediana"],
            "Menor valor": dispersao["menor_valor"], "Maior valor": dispersao["maior_valor"],
            "Amplitude": dispersao["amplitude"], "Desvio padrão": dispersao["desvio_padrao"],
            "Q1": quartis["q1"], "Q3": quartis["q3"], "IQR": quartis["iqr"],
            "Limite inferior": quartis["limite_inferior"], "Limite superior": quartis["limite_superior"],
        }
        for i, moda in enumerate(central["moda"], 1):
            medidas["Moda" if len(central["moda"]) == 1 else f"Moda {i}"] = moda
        for nome, valor in medidas.items():
            resultados.append({"Time": time, "Medida estatística": nome, "Valor": valor, "Unidade": unidade})
        resultados.append({"Time": time, "Medida estatística": "Variância", "Valor": dispersao["variancia"], "Unidade": unidades[1]})
        resultados.append({"Time": time, "Medida estatística": "Quantidade de outliers", "Valor": quartis["quantidade"], "Unidade": "jogos"})
    fig = px.bar(pd.DataFrame(resultados), x="Medida estatística", y="Valor", color="Time",
                 barmode="group", facet_row="Unidade", category_orders={"Unidade": unidades},
                 title=f"Comparação entre equipes — {METADADOS_VARIAVEIS[variavel]['nome']}", text_auto=".2f")
    fig.update_yaxes(matches=None, showticklabels=True)
    fig.update_xaxes(matches=None, showticklabels=True, title_text="Medidas estatísticas", tickangle=-35)
    for row, titulo in zip([3, 2, 1], unidades):
        fig.update_yaxes(title_text=titulo, row=row, col=1)
    fig.update_layout(template="plotly_white", legend_title="Equipe", height=1100)
    return fig

# Função para interpretar os resultados das análises estatísticas de uma equipe
def interpretarMtc(df: pd.DataFrame, time: str):
    dados = filtrarBasePorTime(df, time)
    quartis = quartisOutliers(df, time)
    resultados = []
    for coluna in colunas:
        serie = dados[coluna].dropna()
        meta = METADADOS_VARIAVEIS[coluna]
        modas = ", ".join(f"{v:.2f}" for v in serie.mode())
        tipica = "a mediana" if abs(serie.skew()) >= 0.5 else "a média e a mediana"
        resultados.append({"variavel": meta["nome"], "analise":
            f"Em {len(serie)} partidas, a média foi {serie.mean():.2f} e a mediana {serie.median():.2f} {meta['unidade']}. "
            f"Moda(s): {modas}. Os valores variaram de {serie.min():.2f} a {serie.max():.2f}, "
            f"com desvio padrão amostral de {serie.std(ddof=1):.2f}. O IQR sinalizou {quartis[coluna]['quantidade']} "
            f"ocorrência(s). Considerando a assimetria, {tipica} descreve(m) o valor típico; "
            "as ocorrências atípicas devem ser investigadas no contexto das partidas."})
    return resultados
