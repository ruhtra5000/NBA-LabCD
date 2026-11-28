from flask import Blueprint, render_template, request
from clusterizacao.kmeans import *
from clusterizacao.hierarchical import (
    executarHierarchical,
    categorizarClustersHierarchical,
    graficoDispersaoHierarchical,
    graficoComparacaoKHierarchical,
    graficoComparacaoSilhouetteHierarchical,
)
from clusterizacao.dbscan import (
    CONFIGURACOES_DBSCAN,
    EPS_PADRAO,
    MIN_SAMPLES_PADRAO,
    executarDBSCAN,
    categorizarClustersDBSCAN,
    graficoDispersaoDBSCAN,
    graficoComparacaoKDBSCAN,
    graficoComparacaoSilhouetteDBSCAN,
)
from clusterizacao.gmm import (
    executarGMM,
    categorizarClustersGMM,
    graficoDispersaoGMM,
    graficoComparacaoKGMM,
    graficoComparacaoSilhouetteGMM,
)
from clusterizacao.spectral import (
    executarSpectral,
    categorizarClustersSpectral,
    graficoDispersaoSpectral,
    graficoComparacaoKSpectral,
    graficoComparacaoSilhouetteSpectral,
)

proj2_bp = Blueprint("proj2", __name__)


def _montarContextoClusterizacao(
    k: int,
    resultado,
    modelo,
    categorias,
    fn_dispersao,
    fn_comparacao_k,
    fn_silhouette,
):
    # Gráfico de Dispersão
    eixo_x = request.args.get("eixo_x", "pontos")
    eixo_y = request.args.get("eixo_y", "total_rebotes")

    if eixo_x not in colunas:
        eixo_x = "pontos"

    if eixo_y not in colunas:
        eixo_y = "total_rebotes"

    dispersao_html = fn_dispersao(resultado, modelo, eixo_x, eixo_y).to_html(
        full_html=False,
        include_plotlyjs="cdn"
    )

    # Gráfico Boxplot
    variavel_boxplot = request.args.get("variavel_boxplot", "pontos")
    if variavel_boxplot not in colunas:
        variavel_boxplot = "pontos"

    boxplot_html = graficoBoxplotCluster(resultado, variavel_boxplot).to_html(
        full_html=False,
        include_plotlyjs=False
    )

    # Gráfico de Radar
    radar_html = graficoRadarClusters(resultado).to_html(
        full_html=False,
        include_plotlyjs=False
    )

    # Histograma
    variavel_histograma = request.args.get("variavel_histograma", "pontos")
    if variavel_histograma not in colunas:
        variavel_histograma = "pontos"

    histograma_html = graficoHistogramaClusters(resultado, variavel_histograma).to_html(
        full_html=False,
        include_plotlyjs=False
    )

    # Gráfico de linhas - Evolução
    evolucao_html = graficoEvolucaoClusters(resultado).to_html(
        full_html=False,
        include_plotlyjs=False
    )

    # Gráfico de barras - Qtde partidas
    quantidade_html = graficoQuantidadeClusters(resultado).to_html(
        full_html=False,
        include_plotlyjs=False
    )

    # Comparação da quantidade de partidas alterando K / configuração
    comparacao_k_html = fn_comparacao_k().to_html(
        full_html=False,
        include_plotlyjs=False
    )

    # Comparação do Silhouette Score
    silhouette_html = fn_silhouette().to_html(
        full_html=False,
        include_plotlyjs=False
    )

    return {
        "k": k,
        "categorias": categorias,
        "dispersao": dispersao_html,
        "eixo_x": eixo_x,
        "eixo_y": eixo_y,
        "boxplot": boxplot_html,
        "variavel_boxplot": variavel_boxplot,
        "radar": radar_html,
        "histograma": histograma_html,
        "variavel_histograma": variavel_histograma,
        "evolucao": evolucao_html,
        "quantidade": quantidade_html,
        "comparacao_k": comparacao_k_html,
        "silhouette": silhouette_html,
        "variaveis": colunas,
        "metadados": METADADOS_VARIAVEIS,
    }


# ROTAS
@proj2_bp.route("/proj2")
def projeto2():
    return render_template("projeto2/projeto2.html")


@proj2_bp.route("/kmeans", defaults={"k": 3})
@proj2_bp.route("/kmeans/<int:k>")
def kmeans(k: int):
    k = request.args.get("k", k, type=int)
    resultado, modelo = executarKMeans(k)
    categorias = categorizarClustersKMeans(resultado, modelo)

    contexto = _montarContextoClusterizacao(
        k=k,
        resultado=resultado,
        modelo=modelo,
        categorias=categorias,
        fn_dispersao=graficoDispersao,
        fn_comparacao_k=graficoComparacaoK,
        fn_silhouette=graficoComparacaoSilhouette,
    )
    return render_template("projeto2/kmeans.html", **contexto)


@proj2_bp.route("/hierarchical", defaults={"k": 3})
@proj2_bp.route("/hierarchical/<int:k>")
def hierarchical(k: int):
    k = request.args.get("k", k, type=int)
    resultado, modelo = executarHierarchical(k)
    categorias = categorizarClustersHierarchical(resultado, modelo)

    contexto = _montarContextoClusterizacao(
        k=k,
        resultado=resultado,
        modelo=modelo,
        categorias=categorias,
        fn_dispersao=graficoDispersaoHierarchical,
        fn_comparacao_k=graficoComparacaoKHierarchical,
        fn_silhouette=graficoComparacaoSilhouetteHierarchical,
    )
    return render_template("projeto2/hierarchical.html", **contexto)


@proj2_bp.route("/dbscan", defaults={"k": 3})
@proj2_bp.route("/dbscan/<int:k>")
def dbscan(k: int):
    k = request.args.get("k", k, type=int)
    eps = request.args.get("eps", default=None, type=float)
    min_samples = request.args.get("min_samples", default=None, type=int)

    resultado, modelo = executarDBSCAN(k=k, eps=eps, min_samples=min_samples)
    categorias = categorizarClustersDBSCAN(resultado, modelo)

    contexto = _montarContextoClusterizacao(
        k=k,
        resultado=resultado,
        modelo=modelo,
        categorias=categorias,
        fn_dispersao=graficoDispersaoDBSCAN,
        fn_comparacao_k=graficoComparacaoKDBSCAN,
        fn_silhouette=graficoComparacaoSilhouetteDBSCAN,
    )
    contexto["eps"] = eps if eps is not None else EPS_PADRAO.get(k, 1.35)
    contexto["min_samples"] = min_samples if min_samples is not None else MIN_SAMPLES_PADRAO.get(k, 5)
    contexto["configuracoes_dbscan"] = CONFIGURACOES_DBSCAN

    return render_template("projeto2/dbscan.html", **contexto)


@proj2_bp.route("/gmm", defaults={"k": 3})
@proj2_bp.route("/gmm/<int:k>")
def gmm(k: int):
    k = request.args.get("k", k, type=int)
    resultado, modelo = executarGMM(k)
    categorias = categorizarClustersGMM(resultado, modelo)

    contexto = _montarContextoClusterizacao(
        k=k,
        resultado=resultado,
        modelo=modelo,
        categorias=categorias,
        fn_dispersao=graficoDispersaoGMM,
        fn_comparacao_k=graficoComparacaoKGMM,
        fn_silhouette=graficoComparacaoSilhouetteGMM,
    )
    return render_template("projeto2/gmm.html", **contexto)


@proj2_bp.route("/spectral", defaults={"k": 3})
@proj2_bp.route("/spectral/<int:k>")
def spectral(k: int):
    k = request.args.get("k", k, type=int)
    resultado, modelo = executarSpectral(k)
    categorias = categorizarClustersSpectral(resultado, modelo)

    contexto = _montarContextoClusterizacao(
        k=k,
        resultado=resultado,
        modelo=modelo,
        categorias=categorias,
        fn_dispersao=graficoDispersaoSpectral,
        fn_comparacao_k=graficoComparacaoKSpectral,
        fn_silhouette=graficoComparacaoSilhouetteSpectral,
    )
    return render_template("projeto2/spectral.html", **contexto)


@proj2_bp.route("/teoria")
def teoria():
    return render_template("projeto2/teoria.html")
