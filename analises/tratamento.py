import numpy as np
import pandas as pd
from pathlib import Path
from analises.assimetria import METADADOS_VARIAVEIS

BASE_DIR = Path(__file__).resolve().parent.parent

# Leitura do CSV e tratamento inicial da base
def tratamentoPrevioBase():
    df = pd.read_csv(
            BASE_DIR / "dados" / "stats_times_regular_final.csv",   # Nome da base (.csv)
            parse_dates=["GAME_DATE"]                               # Tratamento de data
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

    # Conversão de valores categóricos para numéricos
    df["resultado"] = df["resultado"].map(lambda valor: {"W": 1, "L": 0}.get(valor, valor))
    for coluna in ["resultado", "vitorias_acumuladas", "derrotas_acumuladas"]:
        df[coluna] = pd.to_numeric(df[coluna], errors="raise")

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

IDENTIFICADORES = {"id_time", "id_partida"}
PERCENTUAIS = {"percentual_vitorias", "percentual_arremessos", "percentual_3_pontos", "percentual_lances_livres"}

# Função para identificar colunas numéricas, excluindo identificadores e a coluna de resultado
def colunasNumericas(df):
    return [c for c in df.select_dtypes(include="number").columns
            if c not in IDENTIFICADORES | {"resultado"}]

# Função para obter metadados de uma coluna, incluindo nome e unidade
def metadadosColuna(coluna):
    if coluna in METADADOS_VARIAVEIS:
        return METADADOS_VARIAVEIS[coluna]
    unidade = "eventos/partida"
    if coluna in PERCENTUAIS:
        unidade = "proporção (0 a 1)"
    elif coluna == "minutos_jogados":
        unidade = "min/partida"
    elif coluna in {"vitorias_acumuladas", "derrotas_acumuladas"}:
        unidade = "jogos acumulados"
    return {"nome": coluna.replace("_", " ").capitalize(), "unidade": unidade}

# Função para analisar valores especiais e inconsistências em um DataFrame
def analisarValoresEspeciais(df):
    regras = []
    invalidos = {c: pd.Series(False, index=df.index) for c in df.columns}

    # Função auxiliar para registrar regras de inconsistência e marcar valores inválidos
    def registrar(nome, mascara, colunas):
        mascara = mascara.fillna(False).astype(bool)
        regras.append({"regra": nome, "quantidade": int(mascara.sum())})
        for c in colunas:
            invalidos[c] |= mascara

    quantitativas = colunasNumericas(df)
    for c in df.select_dtypes(include="number").columns:
        serie = df[c]
        registrar(f"{c}: valor não finito (inf/-inf)", serie.notna() & ~np.isfinite(serie), [c])
        registrar(f"{c}: valor negativo", serie.lt(0), [c])
        if c in PERCENTUAIS:
            registrar(f"{c}: fora de [0, 1]", serie.lt(0) | serie.gt(1), [c])
        else:
            registrar(f"{c}: contagem/código fracionário", serie.notna() & np.isfinite(serie) & serie.mod(1).ne(0), [c])
        if c in IDENTIFICADORES:
            registrar(f"{c}: identificador não positivo", serie.le(0), [c])

    registrar("Resultado fora de {0, 1}", df.resultado.notna() & ~df.resultado.isin([0, 1]), ["resultado"])
    conversoes = df.attrs.get("conversoes_invalidas", {})
    for c, indices in conversoes.items():
        if c in df:
            registrar(f"{c}: conversão inválida", pd.Series(df.index.isin(indices), index=df.index), [c])
    for c in ["nome_time", "confronto"]:
        registrar(f"{c}: texto vazio", df[c].notna() & df[c].astype(str).str.strip().eq(""), [c])

    verificacoes = [
        ("Arremessos convertidos > tentados", "arremessos_convertidos", "arremessos_tentados"),
        ("Cestas de 3 convertidas > tentativas de 3", "cestas_3_pontos_convertidas", "tentativas_3_pontos"),
        ("Cestas de 3 convertidas > total convertido", "cestas_3_pontos_convertidas", "arremessos_convertidos"),
        ("Tentativas de 3 > total de tentativas", "tentativas_3_pontos", "arremessos_tentados"),
        ("Lances livres convertidos > tentados", "lances_livres_convertidos", "lances_livres_tentados"),
    ]
    for nome, a, b in verificacoes:
        registrar(nome, df[a].gt(df[b]), [a, b])
    cs = ["total_rebotes", "rebotes_ofensivos", "rebotes_defensivos"]
    registrar("Rebotes totais diferentes de ofensivos + defensivos",
              df[cs].notna().all(axis=1) & df[cs[0]].ne(df[cs[1]] + df[cs[2]]), cs)
    cs = ["pontos", "arremessos_convertidos", "cestas_3_pontos_convertidas", "lances_livres_convertidos"]
    registrar("Pontos diferentes de 2×FGM + FG3M + FTM",
              df[cs].notna().all(axis=1) & df.pontos.ne(2*df[cs[1]] + df[cs[2]] + df[cs[3]]), cs)

    duplicados_chave = df.duplicated(["id_time", "id_partida"], keep=False)
    registrar("Registros envolvidos em repetição de (time, partida)", duplicados_chave, list(IDENTIFICADORES))
    tabela = []
    for c in df.columns:
        serie = df[c]
        numerica = pd.api.types.is_numeric_dtype(serie)
        outliers = None
        if c in quantitativas:
            validos = serie[serie.notna() & np.isfinite(serie)]
            q1, q3 = validos.quantile([0.25, 0.75], interpolation="linear")
            outliers = int(((validos < q1 - 1.5*(q3-q1)) | (validos > q3 + 1.5*(q3-q1))).sum())
        semantico = ("Identificador nominal" if c in IDENTIFICADORES else "Binária" if c == "resultado"
                     else "Data" if c == "data_partida" else "Categórica/texto" if not numerica else "Numérica")
        tabela.append({"coluna": c, "dtype": str(serie.dtype), "tipo": semantico,
                       "ausentes": int(serie.isna().sum()),
                       "ausentes_pct": 100*serie.isna().mean() if len(df) else 0.0,
                       "zeros": int(serie.eq(0).sum()) if numerica else None,
                       "negativos": int(serie.lt(0).sum()) if numerica else None,
                       "inconsistentes": int(invalidos[c].sum()), "outliers": outliers})
    linhas_inconsistentes = pd.Series(False, index=df.index)
    for m in invalidos.values():
        linhas_inconsistentes |= m
    return {"tabela": tabela, "regras": regras, "linhas": len(df), "colunas": len(df.columns),
            "ausentes": int(df.isna().sum().sum()), "duplicados": int(df.duplicated().sum()),
            "duplicados_chave": int(df.duplicated(["id_time", "id_partida"]).sum()),
            "linhas_inconsistentes": int(linhas_inconsistentes.sum()),
            "zeros": int(df.select_dtypes(include="number").eq(0).sum().sum()),
            "inicio": df.data_partida.min(), "fim": df.data_partida.max()}

# Função para excluir colunas técnicas/textuais da base de dados
def excluirColunasPrevisao(df):
    motivos = {
        "id_time": "Código nominal da equipe. Sua magnitude não representa desempenho; uma futura inclusão exigiria codificação categórica.",
        "id_partida": "Identificador técnico da partida, sem ordem ou distância esportiva interpretável como desempenho.",
        "data_partida": "Data bruta usada para localizar os jogos. Seu uso preditivo exigiria atributos temporais, em vez de tratá-la como uma contagem.",
        "confronto": "Texto que reúne adversário e mando. Essas informações precisariam ser separadas e codificadas antes de entrar em uma matriz numérica.",
        "nome_time": "Nome nominal da equipe, mantido para identificação e filtros. Não é uma medida numérica de desempenho.",
    }
    base = df.drop(columns=list(motivos)).copy()
    return base, [{"coluna": coluna, "motivo": motivo} for coluna, motivo in motivos.items()]
