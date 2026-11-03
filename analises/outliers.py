import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from pyod.models.ecod import ECOD

from analises.tratamento import colunasNumericas

METODOS = {"iqr": "IQR", "zscore": "Z-score", "grubbs": "Grubbs", "ecod": "ECOD",
           "knn": "K-NN", "dbscan": "DBSCAN", "lof": "LOF"}
ALFA = 0.05
CONTAMINACAO = 0.05
K = 5
LOF_K = 20
EPS = 0.5
MIN_SAMPLES = 5

# Função de detecção de outliers em uma série numérica
def detectarSerie(serie):
    serie = pd.to_numeric(serie, errors="coerce")
    serie = serie[serie.notna() & np.isfinite(serie)]
    x = serie.to_numpy(dtype=float)
    n = len(x)
    resultados = {}

    # Função auxiliar para registrar os resultados de cada método
    def registrar(chave, scores, mascara, criterio, limite=None, status="Calculado"):
        resultados[chave] = {"nome": METODOS[chave], "quantidade": int(np.sum(mascara)),
                             "indices": serie.index[np.asarray(mascara, dtype=bool)].tolist(),
                             "scores": pd.Series(scores, index=serie.index),
                             "criterio": criterio, "limite": limite, "status": status, "n": n}

    if n < 3:
        for m in METODOS:
            registrar(m, np.full(n, np.nan), np.zeros(n, dtype=bool), "Requer ao menos 3 valores válidos.",
                      status="Não aplicável: amostra insuficiente")
        return resultados

    q1, q3 = np.quantile(x, [0.25, 0.75], method="linear")
    li, ls = q1 - 1.5*(q3-q1), q3 + 1.5*(q3-q1)
    registrar("iqr", x, (x < li) | (x > ls), f"x < {li:.2f} ou x > {ls:.2f}", (li, ls))
    dp = np.std(x, ddof=1)
    if dp == 0:
        for m in list(METODOS)[1:]:
            registrar(m, np.zeros(n), np.zeros(n, dtype=bool), "Todos os valores são iguais.",
                      status="Não aplicável: série constante")
        return resultados

    z = (x - x.mean()) / dp
    registrar("zscore", z, np.abs(z) > 3, "|z| > 3,00; z = (x − média)/s", 3.0)
    tc = t.ppf(1 - ALFA/(2*n), df=n-2)
    gc = (n-1)/np.sqrt(n) * np.sqrt(tc**2/(n-2+tc**2))
    scores_g = np.abs(z)
    mascara = np.zeros(n, dtype=bool)
    candidato = int(np.argmax(scores_g))
    mascara[candidato] = scores_g[candidato] > gc
    registrar("grubbs", scores_g, mascara, f"Gmáx > {gc:.2f}; α = 0,05; teste bilateral único", float(gc))

    modelo_ecod = ECOD(contamination=CONTAMINACAO, n_jobs=1).fit(x.reshape(-1, 1))
    registrar("ecod", modelo_ecod.decision_scores_, modelo_ecod.labels_ == 1,
              f"Score ECOD > {modelo_ecod.threshold_:.2f}; corte exploratório no percentil 95",
              float(modelo_ecod.threshold_))

    x_padrao = ((x - x.mean()) / np.std(x, ddof=0)).reshape(-1, 1)
    k = min(K, n-1)
    vizinhos = NearestNeighbors(n_neighbors=k+1, metric="euclidean", n_jobs=1).fit(x_padrao)
    distancias = vizinhos.kneighbors(x_padrao)[0][:, -1]  # consulta inclui o próprio ponto
    lim_knn = float(np.quantile(distancias, 1-CONTAMINACAO, method="linear"))
    registrar("knn", distancias, distancias > lim_knn,
              f"Distância ao {k}º vizinho > {lim_knn:.2f}; percentil 95; observação excluída de sua vizinhança", lim_knn)

    rotulos = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES, n_jobs=1).fit_predict(x_padrao)
    registrar("dbscan", rotulos, rotulos == -1, "Rótulo = −1 (ruído); eps = 0,50; min_samples = 5")

    maior_empate = int(serie.value_counts().max())
    k_lof = min(max(LOF_K, maior_empate), n-1)
    lof = LocalOutlierFactor(n_neighbors=k_lof, contamination=CONTAMINACAO, n_jobs=1)
    rotulos_lof = lof.fit_predict(x_padrao)
    registrar("lof", -lof.negative_outlier_factor_, rotulos_lof == -1,
              f"LOF > {-lof.offset_:.2f}; k = {k_lof}; corte exploratório de 5%", float(-lof.offset_))
    return resultados

# Função de análise de outliers por time
def analisarOutliersTime(df, time):
    dados = df[df.nome_time.eq(time)]
    return {c: detectarSerie(dados[c]) for c in colunasNumericas(df)}

# Função de detalhamento dos outliers detectados
def detalharOutliers(df, resultados, coluna, metodo):
    resultado = resultados[coluna][metodo]
    linhas = []
    for indice in resultado["indices"]:
        row = df.loc[indice]
        linhas.append({"data": row.data_partida, "id_partida": row.id_partida,
                       "confronto": row.confronto, "valor": row[coluna],
                       "score": resultado["scores"].loc[indice]})
    return sorted(linhas, key=lambda r: (pd.isna(r["data"]), str(r["data"])))

# Função de interpretação dos resultados de outliers
def interpretarOutliers(resultados, coluna):
    referencia = set(resultados[coluna]["iqr"]["indices"])
    interpretacoes = {}
    for metodo, resultado in resultados[coluna].items():
        indices = set(resultado["indices"])
        if resultado["status"].startswith("Não aplicável"):
            texto = resultado["status"] + ". A quantidade zero não representa uma conclusão de ausência de anomalias."
        else:
            percentual = 100*resultado["quantidade"]/resultado["n"] if resultado["n"] else 0.0
            texto = (f"O método sinalizou {resultado['quantidade']} de {resultado['n']} jogos ({percentual:.2f}%). "
                     f"Dessas ocorrências, {len(indices & referencia)} também foram sinalizadas pelo IQR "
                     f"e {len(indices - referencia)} não foram. ")
            if metodo in {"ecod", "knn", "lof"}:
                texto += "O corte exploratório de 5% depende dos scores e dos empates; não estima a quantidade real de erros."
            elif metodo == "grubbs":
                texto += "Grubbs verifica somente o maior desvio em um teste único, condicionado à normalidade aproximada."
            elif metodo == "dbscan":
                texto += "As sinalizações representam ruído para eps = 0,50 e min_samples = 5 na escala padronizada."
            elif metodo == "zscore":
                texto += "O limite de três desvios amostrais avalia distância à média e pode divergir das cercas baseadas em quartis."
        interpretacoes[metodo] = texto
    return interpretacoes
