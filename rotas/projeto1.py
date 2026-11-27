from functools import lru_cache
from flask import Blueprint, abort, render_template, request
from analises.tratamento import (
    analisarValoresEspeciais, colunasNumericas, metadadosColuna, df
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

proj1_bp = Blueprint("proj1", __name__)

# AUXILIARES
@lru_cache(maxsize=2)
# Função para obter os resultados de outliers para um time específico
def resultadosOutliers(time):
    return analisarOutliersTime(df, time)


# ROTAS
@proj1_bp.route("/proj1")
def projeto1():
    return render_template("projeto1/projeto1.html")

@proj1_bp.route("/mtc/<time>")
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
        "projeto1/mtc.html",
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

@proj1_bp.route("/assimetria/<time>")
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
        "projeto1/assimetria.html",
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

@proj1_bp.route("/cv/<time>")
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
        "projeto1/cv.html",
        time_selecionado=time,
        estatCv=estatCv,
        quadroResumo=quadroResumo,
        graficoEscalas=graficoEscalas,
        graficoComparativo=graficoComparativo,
    )

@proj1_bp.route("/iqr/<time>")
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
        "projeto1/iqr.html",
        time_selecionado=time,
        estatIqr=estatIqr,
        quadroResumo=quadroResumo,
        graficoRobustez=graficoRobustez,
        graficoComparativo=graficoComparativo,
    )

@proj1_bp.route("/outliers/<time>")
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
        "projeto1/outliers.html", time_selecionado=time, variavel_selecionada=variavel,
        variaveis={c: metadadosColuna(c) for c in colunasNumericas(df)},
        metodos=METODOS, resultados=resultados, inconsistencias=inconsistencias,
        interpretacoes=interpretarOutliers(resultados, variavel),
        detalhes={m: detalharOutliers(df, resultados, variavel, m) for m in METODOS},
    )