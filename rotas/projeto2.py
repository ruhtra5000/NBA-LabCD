from flask import Blueprint, render_template, request
from clusterizacao.kmeans import *

proj2_bp = Blueprint("proj2", __name__)


# ROTAS
@proj2_bp.route("/proj2")
def projeto2():
    return render_template("projeto2/projeto2.html")

@proj2_bp.route("/kmeans/<int:k>")
def kmeans(k: int):
    resultado, modelo = executarKMeans(k)

    categorias = categorizarClustersKMeans(resultado, modelo)

    # Gráfico de Dispersão
    eixo_x = request.args.get("eixo_x", "pontos")
    eixo_y = request.args.get("eixo_y", "total_rebotes")

    if eixo_x not in colunas:
        eixo_x = "pontos"

    if eixo_y not in colunas:
        eixo_y = "total_rebotes"

    dispersao = graficoDispersao(resultado, modelo, eixo_x, eixo_y)

    dispersao_html = dispersao.to_html(
        full_html=False,
        include_plotlyjs="cdn"
    )

    # Gráfico Boxplot
    variavel_boxplot = request.args.get("variavel_boxplot", "pontos")

    if variavel_boxplot not in colunas:
        variavel_boxplot = "pontos"

    grafico_boxplot = graficoBoxplotCluster(
        resultado,
        variavel_boxplot
    )

    boxplot_html = grafico_boxplot.to_html(
        full_html=False,
        include_plotlyjs=False
    )

    # Grafico de Radar
    grafico_radar = graficoRadarClusters(resultado)

    radar_html = grafico_radar.to_html(
        full_html=False,
        include_plotlyjs=False
    )

    # Histograma
    variavel_histograma = request.args.get("variavel_histograma", "pontos")

    if variavel_histograma not in colunas:
        variavel_histograma = "pontos"

    grafico_histograma = graficoHistogramaClusters(
        resultado,
        variavel_histograma
    )

    histograma_html = grafico_histograma.to_html(
        full_html=False,
        include_plotlyjs=False
    )

    # Gráfico de linhas - Evolução
    grafico_evolucao = graficoEvolucaoClusters(resultado)

    evolucao_html = grafico_evolucao.to_html(
        full_html=False,
        include_plotlyjs=False
    )

    # Gráfico de barras - Qtde partidas
    grafico_quantidade = graficoQuantidadeClusters(resultado)

    quantidade_html = grafico_quantidade.to_html(
        full_html=False,
        include_plotlyjs=False
    )

    # Comparação da quantidade de partidas, alterando K
    grafico_comparacao_k = graficoComparacaoK()

    comparacao_k_html = grafico_comparacao_k.to_html(
        full_html=False,
        include_plotlyjs=False
    )

    # Comparação do Silhouette Score
    grafico_silhouette = graficoComparacaoSilhouette()

    silhouette_html = grafico_silhouette.to_html(
        full_html=False,
        include_plotlyjs=False
    )

    # Retorno
    return render_template(
        "projeto2/kmeans.html",
        k=k,
        categorias=categorias,
        dispersao=dispersao_html,
        eixo_x=eixo_x,
        eixo_y=eixo_y,
        boxplot=boxplot_html,
        variavel_boxplot=variavel_boxplot,
        radar=radar_html,
        histograma=histograma_html,
        variavel_histograma=variavel_histograma,
        evolucao=evolucao_html,
        quantidade=quantidade_html,
        comparacao_k=comparacao_k_html,
        silhouette=silhouette_html,
        variaveis=colunas,
        metadados=METADADOS_VARIAVEIS,
    )

@proj2_bp.route("/hierarchical")
def hierarchical():
    return render_template("projeto2/hierarchical.html")

@proj2_bp.route("/dbscan")
def dbscan():
    return render_template("projeto2/dbscan.html")

@proj2_bp.route("/gmm")
def gmm():
    return render_template("projeto2/gmm.html")

@proj2_bp.route("/spectral")
def spectral():
    return render_template("projeto2/spectral.html")

@proj2_bp.route("/teoria")
def teoria():
    return render_template("projeto2/teoria.html")
