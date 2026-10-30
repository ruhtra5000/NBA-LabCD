
from flask import Flask, render_template
from analises.tratamento import tratamentoPrevioBase, imprimirQualidadeDados

# Inicializações
app = Flask(__name__)
df = tratamentoPrevioBase()

#imprimirQualidadeDados(df)

# Rotas
@app.route("/")
def inicio():
    return render_template("index.html")

@app.route("/mtc")
def medidasTendCentral():
    return render_template("mtc.html")

@app.route("/assimetria")
def assimetria():
    return render_template("assimetria.html")

@app.route("/cv")
def coefVariacao():
    return render_template("cv.html")

@app.route("/iqr")
def intervInterquartil():
    return render_template("iqr.html")


if __name__ == "__main__":
    app.run(debug=True)
