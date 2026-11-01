import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Colunas analisadas no projeto e seus metadados
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


def classificarRegularidade(cv_pct: float) -> str:
    if cv_pct <= 15.0:
        return "Alta consistência (CV ≤ 15%)"
    elif cv_pct <= 30.0:
        return "Consistência moderada (15% < CV ≤ 30%)"
    return "Alta dispersão / Irregular (CV > 30%)"


def verificarValidadeCv(media: float) -> str:
    if media < 10.0:
        return "Cautela: Média baixa (< 10) infla o CV"
    return "Plenamente válida (escala de razão, média distante de 0)"


def calcularCvTime(df: pd.DataFrame, time: str) -> dict:
    dados_time = filtrarBasePorTime(df, time)
    resultados = {}

    for coluna in colunas:
        serie = dados_time[coluna].dropna()

        media = float(serie.mean())
        # Desvio padrão amostral (ddof=1, denominador n - 1)
        dp_amostral = float(serie.std(ddof=1))

        cv_fracao = (dp_amostral / media) if media != 0 else 0.0
        cv_pct = cv_fracao * 100.0

        # Medidas robustas e assimetria para confronto (Item 6 do CV)
        mediana = float(serie.median())
        q1 = float(serie.quantile(0.25))
        q3 = float(serie.quantile(0.75))
        iqr = q3 - q1
        mad = float(np.median(np.abs(serie - mediana)))

        cv_iqr_mediana = ((iqr / mediana) * 100.0) if mediana != 0 else 0.0
        cv_mad_mediana = ((mad / mediana) * 100.0) if mediana != 0 else 0.0
        g1_fisher = float(serie.skew())

        if abs(g1_fisher) >= 0.50:
            adequacao_cv = "Recomendado usar IQR/Mediana ou MAD/Mediana (assimetria moderada)"
        elif media < 10.0:
            adequacao_cv = "CV válido com cautela (média pequena infla o percentual)"
        else:
            adequacao_cv = "CV clássico (s/x̄) é plenamente adequado"

        resultados[coluna] = {
            "nome_exibicao": METADADOS_VARIAVEIS[coluna]["nome"],
            "unidade": METADADOS_VARIAVEIS[coluna]["unidade"],
            "media": f"{media:.2f}",
            "media_num": round(media, 2),
            "desvio_padrao": f"{dp_amostral:.2f}",
            "dp_num": round(dp_amostral, 2),
            "cv_fracao": f"{cv_fracao:.4f}",
            "cv_pct": f"{cv_pct:.2f}%",
            "cv_pct_num": round(cv_pct, 2),
            "validade": verificarValidadeCv(media),
            "classificacao": classificarRegularidade(cv_pct),
            "mediana": f"{mediana:.2f}",
            "iqr": f"{iqr:.2f}",
            "g1_fisher": f"{g1_fisher:.2f}",
            "g1_num": round(g1_fisher, 2),
            "cv_iqr_mediana": f"{cv_iqr_mediana:.2f}%",
            "cv_mad_mediana": f"{cv_mad_mediana:.2f}%",
            "adequacao_cv": adequacao_cv,
        }

    return resultados


def quadroResumoCv(df: pd.DataFrame) -> dict:
    cavs = calcularCvTime(df, "Cleveland Cavaliers")
    nets = calcularCvTime(df, "Brooklyn Nets")

    quadro = {}
    for coluna in colunas:
        cv_cavs = cavs[coluna]["cv_pct_num"]
        cv_nets = nets[coluna]["cv_pct_num"]
        dif = round(abs(cv_cavs - cv_nets), 2)

        if cv_cavs < cv_nets:
            mais_consistente = "Cleveland Cavaliers"
            mais_irregular = "Brooklyn Nets"
        elif cv_nets < cv_cavs:
            mais_consistente = "Brooklyn Nets"
            mais_irregular = "Cleveland Cavaliers"
        else:
            mais_consistente = "Empate"
            mais_irregular = "Empate"

        quadro[coluna] = {
            "nome_exibicao": METADADOS_VARIAVEIS[coluna]["nome"],
            "unidade_original": METADADOS_VARIAVEIS[coluna]["unidade"],
            "cavs_media_dp": f"{cavs[coluna]['media']} ± {cavs[coluna]['desvio_padrao']}",
            "cavs_cv": cavs[coluna]["cv_pct"],
            "cavs_cv_num": cv_cavs,
            "nets_media_dp": f"{nets[coluna]['media']} ± {nets[coluna]['desvio_padrao']}",
            "nets_cv": nets[coluna]["cv_pct"],
            "nets_cv_num": cv_nets,
            "diferenca_pp": f"{dif:.2f} p.p.",
            "mais_consistente": mais_consistente,
            "mais_irregular": mais_irregular,
        }

    return quadro


def graficoDesvioVsCv(df: pd.DataFrame, time: str):
    estat = calcularCvTime(df, time)
    nomes_eixo = [estat[c]["nome_exibicao"] for c in colunas]
    valores_dp = [estat[c]["dp_num"] for c in colunas]
    valores_cv = [estat[c]["cv_pct_num"] for c in colunas]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Desvio Padrão Bruto (s) — Escalas Incompatíveis",
            "Coeficiente de Variação (CV %) — Escala Adimensional Comparável"
        ),
        horizontal_spacing=0.12
    )

    fig.add_trace(
        go.Bar(
            name="Desvio Padrão Amostral (s)",
            x=nomes_eixo,
            y=valores_dp,
            marker_color="#64748B",
            text=[f"{v:.2f}" for v in valores_dp],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Desvio Padrão (s): %{y:.2f} (unidade bruta)<extra></extra>"
        ),
        row=1,
        col=1
    )

    fig.add_trace(
        go.Bar(
            name="Coeficiente de Variação (CV %)",
            x=nomes_eixo,
            y=valores_cv,
            marker_color="#4F46E5",
            text=[f"{v:.2f}%" for v in valores_cv],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>CV: %{y:.2f}% (adimensional)<extra></extra>"
        ),
        row=1,
        col=2
    )

    fig.add_hline(
        y=15.0,
        line_width=1.5,
        line_dash="dot",
        line_color="#16A34A",
        annotation_text="Alta consistência (≤ 15%)",
        annotation_position="top left",
        row=1,
        col=2
    )
    fig.add_hline(
        y=30.0,
        line_width=1.5,
        line_dash="dash",
        line_color="#D97706",
        annotation_text="Alta dispersão (> 30%)",
        annotation_position="top left",
        row=1,
        col=2
    )

    fig.update_layout(
        title=f"Por que o Desvio Padrão Bruto Engana entre Escalas Distintas? — {time}",
        template="plotly_white",
        height=490,
        showlegend=False
    )

    fig.update_xaxes(title_text="Variáveis Analisadas", tickangle=-20, row=1, col=1)
    fig.update_xaxes(title_text="Variáveis Analisadas", tickangle=-20, row=1, col=2)
    fig.update_yaxes(title_text="Desvio Padrão Amostral s (unidades brutas)", range=[0, max(valores_dp) * 1.22], row=1, col=1)
    fig.update_yaxes(title_text="Coeficiente de Variação CV (%)", range=[0, max(valores_cv) * 1.22], row=1, col=2)

    return fig


def graficoComparativoCvTimes(df: pd.DataFrame):
    cavs = calcularCvTime(df, "Cleveland Cavaliers")
    nets = calcularCvTime(df, "Brooklyn Nets")

    nomes_eixo = [METADADOS_VARIAVEIS[c]["nome"] for c in colunas]
    valores_cavs = [cavs[c]["cv_pct_num"] for c in colunas]
    valores_nets = [nets[c]["cv_pct_num"] for c in colunas]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            name="Cleveland Cavaliers",
            x=nomes_eixo,
            y=valores_cavs,
            marker_color="#860038",
            text=[f"{v:.2f}%" for v in valores_cavs],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Cleveland Cavaliers — CV: %{y:.2f}%<extra></extra>"
        )
    )

    fig.add_trace(
        go.Bar(
            name="Brooklyn Nets",
            x=nomes_eixo,
            y=valores_nets,
            marker_color="#1F2937",
            text=[f"{v:.2f}%" for v in valores_nets],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Brooklyn Nets — CV: %{y:.2f}%<extra></extra>"
        )
    )

    fig.add_hline(
        y=15.0,
        line_width=1.5,
        line_dash="dot",
        line_color="#16A34A",
        annotation_text="Faixa Muito Consistente (≤ 15%)",
        annotation_position="top right"
    )
    fig.add_hline(
        y=30.0,
        line_width=1.5,
        line_dash="dash",
        line_color="#D97706",
        annotation_text="Faixa de Alta Dispersão (> 30%)",
        annotation_position="top right"
    )

    fig.update_layout(
        title="Comparação de Regularidade (CV %) por Variável — Cleveland Cavaliers vs. Brooklyn Nets",
        xaxis_title="Variáveis Analisadas",
        yaxis_title="Coeficiente de Variação CV (%)",
        barmode="group",
        template="plotly_white",
        yaxis=dict(range=[0, max(max(valores_cavs), max(valores_nets)) * 1.2]),
        legend_title="Equipes",
        height=480
    )

    return fig
