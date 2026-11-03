from functools import lru_cache
from math import isfinite
from flask import Flask, abort, render_template, request
from analises.tratamento import (
    tratamentoPrevioBase, imprimirQualidadeDados, analisarValoresEspeciais,
    excluirColunasPrevisao, colunasNumericas, metadadosColuna,
)
from analises.outliers import METODOS, analisarOutliersTime, detalharOutliers, interpretarOutliers
from analises.mtc import *
from analises.assimetria import (
    METADADOS_VARIAVEIS,
    calcularAssimetria,
    compararAssimetriaTimes,
    graficoHistKdeBoxplot,
    graficoComparativoAssimetria,
)
from analises.cv import (
    calcularCvTime,
    quadroResumoCv,
    graficoDesvioVsCv,
    graficoComparativoCvTimes,
)
from analises.iqr import (
    calcularIqrTime,
    quadroResumoIqr,
    graficoIqrVsAmplitudeDesvio,
    graficoComparativoIqrTimes,
)

# Inicializações
app = Flask(__name__)
df = tratamentoPrevioBase()

#imprimirQualidadeDados(df)

@app.template_filter("numero")
# Função para formatar números com duas casas decimais, tratando valores None e não finitos
def formatarNumero(valor):
    if valor is None:
        return "Não aplicável"
    try:
        return f"{float(valor):.2f}" if isfinite(float(valor)) else "Não definido"
    except (ValueError, TypeError):
        return "Não definido"

@lru_cache(maxsize=2)
# Função para obter os resultados de outliers para um time específico
def resultadosOutliers(time):
    return analisarOutliersTime(df, time)

# Rotas
@app.route("/")
def inicio():
    qualidade = analisarValoresEspeciais(df)
    base_previsao, colunas_excluidas = excluirColunasPrevisao(df)
    return render_template("index.html", qualidade=qualidade,
                           tipos_dados={c: str(df[c].dtype) for c in df.columns},
                           colunas_excluidas=colunas_excluidas,
                           quantidade_colunas_previsao=len(base_previsao.columns))

@app.route("/mtc/<time>")
def medidasTendCentral(time: str):
    variavel_selecionada = request.args.get("variavel", "pontos")
    if variavel_selecionada not in METADADOS_VARIAVEIS:
        variavel_selecionada = "pontos"

    estatMediaModaMediana = mediaModaMediana(df, time)
    estatQuartisOutliers = quartisOutliers(df, time)
    estatMedidasDispersao = medidasDispersao(df, time)

    listaValOutliers = listaOutliers(estatQuartisOutliers, METADADOS_VARIAVEIS)

    boxplotGrafico = boxplotMtc(df, time).to_html(
        full_html=False,
        include_plotlyjs="cdn"
    )

    graficoComparativo = graficoComparacao(df, variavel_selecionada).to_html(
        full_html=False,
        include_plotlyjs="cdn"
    )

    return render_template(
        "mtc.html",
        time_selecionado=time,
        variavel_selecionada=variavel_selecionada,
        estatMediaModaMediana = estatMediaModaMediana,
        estatQuartisOutliers = estatQuartisOutliers,
        estatMedidasDispersao = estatMedidasDispersao,
        listaValOutliers = listaValOutliers,
        boxplotGrafico = boxplotGrafico,
        graficoComparativo = graficoComparativo,
        metadados_variaveis = METADADOS_VARIAVEIS,
        interpretacoes=interpretarMtc(df, time),
    )

@app.route("/assimetria/<time>")
def assimetria(time: str):
    variavel_selecionada = request.args.get("variavel", "pontos")
    if variavel_selecionada not in METADADOS_VARIAVEIS:
        variavel_selecionada = "pontos"

    estatAssimetria = calcularAssimetria(df, time)
    comparativoTimes = compararAssimetriaTimes(df)

    graficoDistribuicao = graficoHistKdeBoxplot(df, time, variavel_selecionada).to_html(
        full_html=False,
        include_plotlyjs="cdn"
    )
    graficoComparativo = graficoComparativoAssimetria(df).to_html(
        full_html=False,
        include_plotlyjs=False
    )

    return render_template(
        "assimetria.html",
        time_selecionado=time,
        variavel_selecionada=variavel_selecionada,
        metadados_variaveis=METADADOS_VARIAVEIS,
        estatAssimetria=estatAssimetria,
        comparativoTimes=comparativoTimes,
        graficoDistribuicao=graficoDistribuicao,
        graficoComparativo=graficoComparativo,
        detalhe_cavs=calcularAssimetria(df, "Cleveland Cavaliers"),
        quartis_cavs=calcularIqrTime(df, "Cleveland Cavaliers"),
        quartis_nets=calcularIqrTime(df, "Brooklyn Nets"),
    )

@app.route("/cv/<time>")
def coefVariacao(time: str):
    estatCv = calcularCvTime(df, time)
    quadroResumo = quadroResumoCv(df)

    graficoEscalas = graficoDesvioVsCv(df, time).to_html(
        full_html=False,
        include_plotlyjs="cdn"
    )
    graficoComparativo = graficoComparativoCvTimes(df).to_html(
        full_html=False,
        include_plotlyjs=False
    )

    return render_template(
        "cv.html",
        time_selecionado=time,
        estatCv=estatCv,
        quadroResumo=quadroResumo,
        graficoEscalas=graficoEscalas,
        graficoComparativo=graficoComparativo,
    )

@app.route("/iqr/<time>")
def intervInterquartil(time: str):
    estatIqr = calcularIqrTime(df, time)
    quadroResumo = quadroResumoIqr(df)

    graficoRobustez = graficoIqrVsAmplitudeDesvio(df, time).to_html(
        full_html=False,
        include_plotlyjs="cdn"
    )
    graficoComparativo = graficoComparativoIqrTimes(df).to_html(
        full_html=False,
        include_plotlyjs=False
    )

    return render_template(
        "iqr.html",
        time_selecionado=time,
        estatIqr=estatIqr,
        quadroResumo=quadroResumo,
        graficoRobustez=graficoRobustez,
        graficoComparativo=graficoComparativo,
    )

@app.route("/outliers/<time>")
# Função para exibir os valores inconsistentes e outliers de um time específico
def valoresInconsistentesOutliers(time: str):
    if time not in {"Cleveland Cavaliers", "Brooklyn Nets"}:
        abort(404)
    variavel = request.args.get("variavel", "pontos")
    if variavel not in colunasNumericas(df):
        variavel = "pontos"
    resultados = resultadosOutliers(time)
    qualidade = analisarValoresEspeciais(df[df["nome_time"] == time])
    inconsistencias = {r["coluna"]: r["inconsistentes"] for r in qualidade["tabela"]}
    return render_template(
        "outliers.html", time_selecionado=time, variavel_selecionada=variavel,
        variaveis={c: metadadosColuna(c) for c in colunasNumericas(df)},
        metodos=METODOS, resultados=resultados, inconsistencias=inconsistencias,
        interpretacoes=interpretarOutliers(resultados, variavel),
        detalhes={m: detalharOutliers(df, resultados, variavel, m) for m in METODOS},
    )


if __name__ == "__main__":
    app.run(debug=True)
