
from flask import Flask, render_template
from analises.tratamento import tratamentoPrevioBase, imprimirQualidadeDados
from analises.mtc import *

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

    return render_template(
        "mtc.html",
        time_selecionado=time,
        estatMediaModaMediana = estatMediaModaMediana,
        estatQuartisOutliers = estatQuartisOutliers
    )

@app.route("/assimetria/<time>")
def assimetria(time: str):
    return render_template(
        "assimetria.html",
        time_selecionado=time,
    )

@app.route("/cv/<time>")
def coefVariacao(time: str):
    return render_template(
        "cv.html",
        time_selecionado=time,
    )

@app.route("/iqr/<time>")
def intervInterquartil(time: str):
    return render_template(
        "iqr.html",
        time_selecionado=time,
    )


if __name__ == "__main__":
    app.run(debug=True)
