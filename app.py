
from flask import Flask, render_template, request
from analises.tratamento import tratamentoPrevioBase, imprimirQualidadeDados
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

# Inicializações
app = Flask(__name__)
df = tratamentoPrevioBase()

#imprimirQualidadeDados(df)

# Rotas
@app.route("/")
def inicio():
    return render_template("index.html")

@app.route("/mtc/<time>")
def medidasTendCentral(time: str):
    estatMediaModaMediana = mediaModaMediana(df, time)
    estatQuartisOutliers = quartisOutliers(df, time)
    estatMedidasDispersao = medidasDispersao(df, time)

    boxplotGrafico = boxplotMtc(df, time).to_html(
        full_html=False,
        include_plotlyjs="cdn"
    )

    return render_template(
        "mtc.html",
        time_selecionado=time,
        estatMediaModaMediana = estatMediaModaMediana,
        estatQuartisOutliers = estatQuartisOutliers,
        estatMedidasDispersao = estatMedidasDispersao,
        boxplotGrafico = boxplotGrafico
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
    return render_template(
        "iqr.html",
        time_selecionado=time,
    )


if __name__ == "__main__":
    app.run(debug=True)
