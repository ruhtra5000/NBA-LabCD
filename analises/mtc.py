import pandas as pd

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