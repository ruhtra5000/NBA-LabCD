
from flask import Flask, render_template

app = Flask(__name__)


@app.route("/")
def inicio():
    return render_template("index.html")


@app.route("/sobre")
def sobre():
    return "Esta é a página sobre do meu projeto."


if __name__ == "__main__":
    app.run(debug=True)
