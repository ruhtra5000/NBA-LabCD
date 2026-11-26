from math import isfinite

from flask import Flask, render_template, request
from analises.tratamento import (
    analisarValoresEspeciais, excluirColunasPrevisao, df
)
from rotas.projeto1 import proj1_bp
from rotas.projeto2 import proj2_bp


# INICIALIZAÇÕES
app = Flask(__name__)

#imprimirQualidadeDados(df)


# AUXILIARES
@app.template_filter("numero")
# Função para formatar números com duas casas decimais, tratando valores None e não finitos
def formatarNumero(valor):
    if valor is None:
        return "Não aplicável"
    try:
        return f"{float(valor):.2f}" if isfinite(float(valor)) else "Não definido"
    except (ValueError, TypeError):
        return "Não definido"


# ROTAS
@app.route("/")
def inicio():
    qualidade = analisarValoresEspeciais(df)
    base_previsao, colunas_excluidas = excluirColunasPrevisao(df)
    return render_template("index.html", qualidade=qualidade,
                           tipos_dados={c: str(df[c].dtype) for c in df.columns},
                           colunas_excluidas=colunas_excluidas,
                           quantidade_colunas_previsao=len(base_previsao.columns))

# Carregamento de blueprints (diretório "rotas")
app.register_blueprint(proj1_bp)
app.register_blueprint(proj2_bp)

if __name__ == "__main__":
    app.run(debug=True)
