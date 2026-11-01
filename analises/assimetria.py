import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Colunas analisadas no projeto e seus metadados (nome legível e unidade)
colunas = [
    "pontos",
    "total_rebotes",
    "assistencias",
    "roubos_bola",
    "bloqueios",
    "faltas_pessoais",
    "cestas_3_pontos_convertidas"
]

METADADOS_VARIAVEIS = {
    "pontos": {
        "nome": "Pontos (PTS)",
        "unidade": "pts/partida"
    },
    "total_rebotes": {
        "nome": "Total de Rebotes (REB)",
        "unidade": "reb/partida"
    },
    "assistencias": {
        "nome": "Assistências (AST)",
        "unidade": "ast/partida"
    },
    "roubos_bola": {
        "nome": "Roubos de Bola (STL)",
        "unidade": "stl/partida"
    },
    "bloqueios": {
        "nome": "Bloqueios (BLK)",
        "unidade": "blk/partida"
    },
    "faltas_pessoais": {
        "nome": "Faltas Pessoais (PF)",
        "unidade": "faltas/partida"
    },
    "cestas_3_pontos_convertidas": {
        "nome": "Cestas de 3 Pontos (FG3M)",
        "unidade": "cestas/partida"
    }
}


def filtrarBasePorTime(df: pd.DataFrame, time: str) -> pd.DataFrame:
    return df[df["nome_time"] == time]


def classificarDirecao(g1: float) -> str:
    if abs(g1) < 0.05:
        return "Simétrica"
    elif g1 > 0:
        return "Assimétrica à direita (Positiva)"
    return "Assimétrica à esquerda (Negativa)"


def classificarIntensidade(g1: float) -> str:
    abs_g1 = abs(g1)
    if abs_g1 < 0.50:
        return "Aproximadamente simétrica"
    elif abs_g1 < 1.00:
        return "Moderada"
    return "Forte"


def calcularAssimetria(df: pd.DataFrame, time: str) -> dict:
    dados_time = filtrarBasePorTime(df, time)
    resultados = {}

    for coluna in colunas:
        serie = dados_time[coluna].dropna()
        media = float(serie.mean())
        mediana = float(serie.median())
        dp = float(serie.std(ddof=1))
        q1 = float(serie.quantile(0.25))
        q3 = float(serie.quantile(0.75))
        iqr = q3 - q1

        # Coeficiente ajustado de Fisher-Pearson (G1) - padrão do pandas.Series.skew()
        g1_fisher = float(serie.skew())

        # Coeficiente quartílico de Bowley
        bowley = float(((q3 - mediana) - (mediana - q1)) / iqr) if iqr != 0 else 0.0

        # 2º Coeficiente de Pearson: 3 * (Média - Mediana) / s
        pearson = float((3.0 * (media - mediana)) / dp) if dp != 0 else 0.0

        direcao = classificarDirecao(g1_fisher)
        intensidade = classificarIntensidade(g1_fisher)

        if abs(g1_fisher) >= 0.50:
            medida_recomendada = "Mediana (assimetria moderada/forte)"
        else:
            medida_recomendada = "Média ou Mediana (distribuição aprox. simétrica)"

        resultados[coluna] = {
            "nome_exibicao": METADADOS_VARIAVEIS[coluna]["nome"],
            "unidade": METADADOS_VARIAVEIS[coluna]["unidade"],
            "media": f"{media:.2f}",
            "mediana": f"{mediana:.2f}",
            "g1_fisher": f"{g1_fisher:.2f}",
            "g1_fisher_num": round(g1_fisher, 2),
            "pearson": f"{pearson:.2f}",
            "bowley": f"{bowley:.2f}",
            "direcao": direcao,
            "intensidade": intensidade,
            "medida_recomendada": medida_recomendada
        }

    return resultados


def compararAssimetriaTimes(df: pd.DataFrame) -> dict:
    cavs = calcularAssimetria(df, "Cleveland Cavaliers")
    nets = calcularAssimetria(df, "Brooklyn Nets")

    comparativo = {}
    for coluna in colunas:
        g1_cavs = cavs[coluna]["g1_fisher_num"]
        g1_nets = nets[coluna]["g1_fisher_num"]
        mesmo_sinal = (g1_cavs >= 0 and g1_nets >= 0) or (g1_cavs < 0 and g1_nets < 0)

        if abs(g1_cavs) > abs(g1_nets):
            mais_assimetrica = "Cleveland Cavaliers"
        elif abs(g1_nets) > abs(g1_cavs):
            mais_assimetrica = "Brooklyn Nets"
        else:
            mais_assimetrica = "Empate"

        comparativo[coluna] = {
            "nome_exibicao": METADADOS_VARIAVEIS[coluna]["nome"],
            "unidade": METADADOS_VARIAVEIS[coluna]["unidade"],
            "cavs_g1": cavs[coluna]["g1_fisher"],
            "cavs_direcao": cavs[coluna]["direcao"],
            "cavs_intensidade": cavs[coluna]["intensidade"],
            "nets_g1": nets[coluna]["g1_fisher"],
            "nets_direcao": nets[coluna]["direcao"],
            "nets_intensidade": nets[coluna]["intensidade"],
            "padrao_repete": "Sim (mesma direção)" if mesmo_sinal else "Não (direções opostas)",
            "mais_assimetrica": mais_assimetrica
        }

    return comparativo


def _calcularKdeGaussiano(valores: np.ndarray, num_pontos: int = 200):
    n = len(valores)
    dp = np.std(valores, ddof=1)
    if n < 2 or dp == 0:
        x_grid = np.linspace(valores.min() - 1, valores.max() + 1, num_pontos)
        return x_grid, np.zeros_like(x_grid)

    # Regra de Scott para largura de banda (bandwidth)
    iqr = np.percentile(valores, 75) - np.percentile(valores, 25)
    sigma_robusto = min(dp, iqr / 1.34) if iqr > 0 else dp
    bandwidth = 1.06 * sigma_robusto * (n ** (-0.2))

    margem = max(dp * 0.8, 1.5)
    x_grid = np.linspace(valores.min() - margem, valores.max() + margem, num_pontos)

    # Cálculo vetorizado do Kernel Gaussiano
    u = (x_grid[:, None] - valores[None, :]) / bandwidth
    densidade = np.mean(np.exp(-0.5 * (u ** 2)) / np.sqrt(2.0 * np.pi), axis=1) / bandwidth
    return x_grid, densidade


def graficoHistKdeBoxplot(df: pd.DataFrame, time: str, variavel: str):
    if variavel not in colunas:
        variavel = colunas[0]

    dados_time = filtrarBasePorTime(df, time)
    serie = dados_time[variavel].dropna()
    valores = serie.to_numpy(dtype=float)

    meta = METADADOS_VARIAVEIS[variavel]
    nome_var = meta["nome"]
    unidade = meta["unidade"]

    media = float(serie.mean())
    mediana = float(serie.median())
    moda = serie.mode().tolist()
    q1 = float(serie.quantile(0.25))
    q3 = float(serie.quantile(0.75))
    iqr = q3 - q1
    lim_inf = q1 - 1.5 * iqr
    lim_sup = q3 + 1.5 * iqr
    g1 = float(serie.skew())

    outliers = serie[(serie < lim_inf) | (serie > lim_sup)]
    valores_validos = serie[(serie >= lim_inf) & (serie <= lim_sup)]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.28, 0.72],
        vertical_spacing=0.08,
        subplot_titles=(
            f"Boxplot — {nome_var} ({unidade})",
            f"Histograma de Densidade + Curva KDE — G₁ = {g1:.2f} ({classificarDirecao(g1)})"
        )
    )

    # 1. Boxplot Horizontal no topo
    fig.add_trace(
        go.Box(
            name="Boxplot",
            y=[nome_var],
            q1=[q1],
            median=[mediana],
            q3=[q3],
            lowerfence=[ float(valores_validos.min()) if not valores_validos.empty else q1 ],
            upperfence=[ float(valores_validos.max()) if not valores_validos.empty else q3 ],
            orientation="h",
            marker_color="#4F46E5",
            boxpoints=False,
            showlegend=True
        ),
        row=1,
        col=1
    )

    # Marcador da Média no Boxplot
    fig.add_trace(
        go.Scatter(
            x=[media],
            y=[nome_var],
            mode="markers",
            name=f"Média ({media:.2f})",
            marker=dict(symbol="diamond", size=11, color="#DC2626"),
            hovertemplate="Média: %{x:.2f} " + unidade + "<extra></extra>"
        ),
        row=1,
        col=1
    )

    # Marcador da Moda no Boxplot
    if moda:
        fig.add_trace(
            go.Scatter(
                x=moda,
                y=[nome_var] * len(moda),
                mode="markers",
                name=f"Moda ({', '.join(str(int(m)) if float(m).is_integer() else f'{m:.2f}' for m in moda)})",
                marker=dict(symbol="triangle-up", size=11, color="#16A34A"),
                hovertemplate="Moda: %{x:.2f} " + unidade + "<extra></extra>"
            ),
            row=1,
            col=1
        )

    # Outliers no Boxplot
    if not outliers.empty:
        fig.add_trace(
            go.Scatter(
                x=outliers,
                y=[nome_var] * len(outliers),
                mode="markers",
                name=f"Outliers ({len(outliers)})",
                marker=dict(symbol="circle", size=8, color="#111827"),
                hovertemplate="Outlier: %{x:.2f} " + unidade + "<extra></extra>"
            ),
            row=1,
            col=1
        )

    # 2. Histograma de Densidade na base
    fig.add_trace(
        go.Histogram(
            x=valores,
            histnorm="probability density",
            name="Histograma (Densidade)",
            marker=dict(
                color="rgba(79, 70, 229, 0.35)",
                line=dict(color="#4F46E5", width=1.2)
            ),
            hovertemplate="Faixa: %{x} " + unidade + "<br>Densidade: %{y:.4f}<extra></extra>"
        ),
        row=2,
        col=1
    )

    # 3. Curva KDE Gaussiana sobreposta
    x_kde, y_kde = _calcularKdeGaussiano(valores)
    fig.add_trace(
        go.Scatter(
            x=x_kde,
            y=y_kde,
            mode="lines",
            name="Curva de Densidade (KDE)",
            line=dict(color="#EA580C", width=3),
            hovertemplate="Valor: %{x:.2f} " + unidade + "<br>KDE: %{y:.4f}<extra></extra>"
        ),
        row=2,
        col=1
    )

    # Linhas verticais de Média e Mediana no Histograma/KDE
    max_y = float(np.max(y_kde)) * 1.15 if len(y_kde) > 0 else 0.1
    fig.add_trace(
        go.Scatter(
            x=[media, media],
            y=[0, max_y],
            mode="lines",
            name=f"Linha da Média ({media:.2f})",
            line=dict(color="#DC2626", width=2, dash="dash"),
            hoverinfo="skip"
        ),
        row=2,
        col=1
    )

    fig.add_trace(
        go.Scatter(
            x=[mediana, mediana],
            y=[0, max_y],
            mode="lines",
            name=f"Linha da Mediana ({mediana:.2f})",
            line=dict(color="#0D9488", width=2, dash="dot"),
            hoverinfo="skip"
        ),
        row=2,
        col=1
    )

    fig.update_layout(
        title=f"Confirmação Visual de Assimetria — {nome_var} ({time})",
        height=560,
        template="plotly_white",
        legend_title="Elementos do Gráfico",
        bargap=0.06
    )

    fig.update_xaxes(title_text=f"{nome_var} ({unidade})", row=2, col=1)
    fig.update_yaxes(title_text="", row=1, col=1)
    fig.update_yaxes(title_text="Densidade de Probabilidade", row=2, col=1)

    return fig


def graficoComparativoAssimetria(df: pd.DataFrame):
    cavs = calcularAssimetria(df, "Cleveland Cavaliers")
    nets = calcularAssimetria(df, "Brooklyn Nets")

    nomes_eixo = [METADADOS_VARIAVEIS[c]["nome"] for c in colunas]
    valores_cavs = [cavs[c]["g1_fisher_num"] for c in colunas]
    valores_nets = [nets[c]["g1_fisher_num"] for c in colunas]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            name="Cleveland Cavaliers",
            x=nomes_eixo,
            y=valores_cavs,
            marker_color="#860038",
            text=[f"{v:.2f}" for v in valores_cavs],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Cleveland Cavaliers — G₁: %{y:.2f} (adimensional)<extra></extra>"
        )
    )

    fig.add_trace(
        go.Bar(
            name="Brooklyn Nets",
            x=nomes_eixo,
            y=valores_nets,
            marker_color="#1F2937",
            text=[f"{v:.2f}" for v in valores_nets],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Brooklyn Nets — G₁: %{y:.2f} (adimensional)<extra></extra>"
        )
    )

    # Linhas de referência para simetria (0) e limiar de assimetria moderada (+0.5 e -0.5)
    fig.add_hline(y=0, line_width=1.5, line_color="#64748B")
    fig.add_hline(
        y=0.5,
        line_width=1.5,
        line_dash="dash",
        line_color="#D97706",
        annotation_text="Limiar Moderado (+0,50)",
        annotation_position="top right"
    )
    fig.add_hline(
        y=-0.5,
        line_width=1.5,
        line_dash="dash",
        line_color="#D97706",
        annotation_text="Limiar Moderado (-0,50)",
        annotation_position="bottom right"
    )

    fig.update_layout(
        title="Comparação do Coeficiente de Assimetria (G₁ de Fisher-Pearson) — Cavaliers vs. Nets",
        xaxis_title="Variáveis Analisadas",
        yaxis_title="Coeficiente de Assimetria G₁ (adimensional)",
        barmode="group",
        template="plotly_white",
        yaxis=dict(range=[-0.75, 0.85]),
        legend_title="Equipes",
        height=480
    )

    return fig
