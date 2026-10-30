import pandas as pd

# Leitura do CSV e tratamento inicial da base
def tratamentoPrevioBase():
    df = pd.read_csv(
            "dados/stats_times_regular_final.csv",  # Nome da base (.csv)
            parse_dates=["GAME_DATE"]               # Tratamento de data
        )

    # Renomeação de colunas
    df = df.rename(columns={
        "Team_ID": "id_time",
        "Game_ID": "id_partida",
        "GAME_DATE": "data_partida",
        "MATCHUP": "confronto",
        "WL": "resultado",
        "W": "vitorias_acumuladas",
        "L": "derrotas_acumuladas",
        "W_PCT": "percentual_vitorias",
        "MIN": "minutos_jogados",
        "FGM": "arremessos_convertidos",
        "FGA": "arremessos_tentados",
        "FG_PCT": "percentual_arremessos",
        "FG3M": "cestas_3_pontos_convertidas",
        "FG3A": "tentativas_3_pontos",
        "FG3_PCT": "percentual_3_pontos",
        "FTM": "lances_livres_convertidos",
        "FTA": "lances_livres_tentados",
        "FT_PCT": "percentual_lances_livres",
        "OREB": "rebotes_ofensivos",
        "DREB": "rebotes_defensivos",
        "REB": "total_rebotes",
        "AST": "assistencias",
        "STL": "roubos_bola",
        "BLK": "bloqueios",
        "TOV": "perdas_bola",
        "PF": "faltas_pessoais",
        "PTS": "pontos",
        "TEAM_NAME": "nome_time"
    })

    return df

# Impressão de informações sobre a qualidade dos dados da base
def imprimirQualidadeDados(df: pd.DataFrame):
    # Calculo de variáveis e constantes
    colunas_percentuais = [
        "percentual_vitorias",
        "percentual_arremessos",
        "percentual_3_pontos",
        "percentual_lances_livres"
    ]

    colunas_numericas = [
        "pontos",
        "assistencias",
        "rebotes_ofensivos",
        "rebotes_defensivos",
        "total_rebotes",
        "roubos_bola",
        "bloqueios",
        "perdas_bola",
        "faltas_pessoais",
        "arremessos_convertidos",
        "arremessos_tentados"
    ]

    erro_arremessos = df[df["arremessos_convertidos"] > df["arremessos_tentados"]]

    erro_tres_pontos = df[df["cestas_3_pontos_convertidas"] > df["arremessos_convertidos"]]

    erro_lances_livres = df[df["lances_livres_convertidos"] > df["lances_livres_tentados"]]

    erro_rebotes = df[df["total_rebotes"] != (df["rebotes_ofensivos"] + df["rebotes_defensivos"])]

    # Impressões
    print(f"\nValores ausentes por coluna:\n{df.isnull().sum()}")

    print(f"\nPercentual de valores ausentes por coluna:\n{(df.isnull().mean() * 100).round(2)}%")

    print(f"\nQuantidade de registros duplicados: {df.duplicated().sum()}")

    print(f"\nInconsistencias por coluna percentual (fora do intervalo [0, 1]):")
    for coluna in colunas_percentuais:
        inconsistentes = df[(df[coluna] < 0) | (df[coluna] > 1)]
        print(f" - {coluna}: {len(inconsistentes)} inconsistências")

    print(f"\nValores negativos por coluna numerica:")
    for coluna in colunas_numericas:
        negativos = df[df[coluna] < 0]
        print(f" - {coluna}: {len(negativos)} valores negativos")

    print(f"\nErros nos arremessos: {len(erro_arremessos)}")

    print(f"\nErros nas cestas de 3 pontos: {len(erro_tres_pontos)}")

    print(f"\nErros nos lances livres: {len(erro_lances_livres)}")

    print(f"\nErros nos rebotes: {len(erro_rebotes)}\n")

    print(df.info())

