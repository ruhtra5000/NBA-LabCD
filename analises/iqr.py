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


def calcularIqrTime(df: pd.DataFrame, time: str) -> dict:
    dados_time = filtrarBasePorTime(df, time)
    resultados = {}

    for coluna in colunas:
        serie = dados_time[coluna].dropna()

        # Quartis via interpolação linear (padrão numpy/pandas, consistente com mtc.py)
        q1 = float(serie.quantile(0.25, interpolation="linear"))
        mediana = float(serie.quantile(0.50, interpolation="linear"))
        q3 = float(serie.quantile(0.75, interpolation="linear"))
        iqr = q3 - q1

        menor = float(serie.min())
        maior = float(serie.max())
        amplitude = maior - menor
        dp = float(serie.std(ddof=1))
        media = float(serie.mean())
        cv_pct = (dp / media * 100.0) if media != 0 else 0.0
        g1 = float(serie.skew())

        # Razões de sensibilidade a valores extremos
        razao_amp_iqr = (amplitude / iqr) if iqr != 0 else 0.0
        razao_dp_iqr = (dp / iqr) if iqr != 0 else 0.0

        # Limites de Tukey moderados (1,5 × IQR) e severos (3,0 × IQR)
        lim_inf_15 = q1 - 1.5 * iqr
        lim_sup_15 = q3 + 1.5 * iqr
        outliers_15 = sorted(serie[(serie < lim_inf_15) | (serie > lim_sup_15)].tolist())

        lim_inf_30 = q1 - 3.0 * iqr
        lim_sup_30 = q3 + 3.0 * iqr
        outliers_30 = sorted(serie[(serie < lim_inf_30) | (serie > lim_sup_30)].tolist())

        outliers_15_str = (
            ", ".join(str(int(v)) if float(v).is_integer() else f"{v:.2f}" for v in outliers_15)
            if outliers_15 else "Nenhum (0)"
        )
        outliers_30_str = (
            ", ".join(str(int(v)) if float(v).is_integer() else f"{v:.2f}" for v in outliers_30)
            if outliers_30 else "Nenhum (0)"
        )

        unidade = METADADOS_VARIAVEIS[coluna]["unidade"]
        interpretacao_faixa = (
            f"50% central dos jogos ficou entre {q1:.2f} e {q3:.2f} (largura de {iqr:.2f} {unidade})"
        )

        resultados[coluna] = {
            "nome_exibicao": METADADOS_VARIAVEIS[coluna]["nome"],
            "unidade": unidade,
            "q1": f"{q1:.2f}",
            "mediana": f"{mediana:.2f}",
            "q3": f"{q3:.2f}",
            "iqr": f"{iqr:.2f}",
            "iqr_num": round(iqr, 2),
            "menor": f"{menor:.2f}",
            "maior": f"{maior:.2f}",
            "amplitude": f"{amplitude:.2f}",
            "amplitude_num": round(amplitude, 2),
            "desvio_padrao": f"{dp:.2f}",
            "dp_num": round(dp, 2),
            "razao_amp_iqr": f"{razao_amp_iqr:.2f}×",
            "razao_dp_iqr": f"{razao_dp_iqr:.2f}",
            "lim_inf_15": f"{lim_inf_15:.2f}",
            "lim_sup_15": f"{lim_sup_15:.2f}",
            "qtd_outliers_15": len(outliers_15),
            "outliers_15_str": outliers_15_str,
            "lim_inf_30": f"{lim_inf_30:.2f}",
            "lim_sup_30": f"{lim_sup_30:.2f}",
            "qtd_outliers_30": len(outliers_30),
            "outliers_30_str": outliers_30_str,
            "interpretacao_faixa": interpretacao_faixa,
            "cv_pct": f"{cv_pct:.2f}%",
            "g1": f"{g1:.2f}",
        }

    return resultados


def quadroResumoIqr(df: pd.DataFrame) -> dict:
    cavs = calcularIqrTime(df, "Cleveland Cavaliers")
    nets = calcularIqrTime(df, "Brooklyn Nets")

    quadro = {}
    for coluna in colunas:
        iqr_cavs = cavs[coluna]["iqr_num"]
        iqr_nets = nets[coluna]["iqr_num"]
        dif = round(abs(iqr_cavs - iqr_nets), 2)

        if iqr_cavs < iqr_nets:
            mais_concentrado = "Cleveland Cavaliers"
            mais_espalhado = "Brooklyn Nets"
        elif iqr_nets < iqr_cavs:
            mais_concentrado = "Brooklyn Nets"
            mais_espalhado = "Cleveland Cavaliers"
        else:
            mais_concentrado = "Empate (Mesmo IQR)"
            mais_espalhado = "Empate (Mesmo IQR)"

        quadro[coluna] = {
            "nome_exibicao": METADADOS_VARIAVEIS[coluna]["nome"],
            "unidade": METADADOS_VARIAVEIS[coluna]["unidade"],
            "cavs_iqr": cavs[coluna]["iqr"],
            "cavs_faixa": f"[{cavs[coluna]['q1']} – {cavs[coluna]['q3']}]",
            "cavs_cv": cavs[coluna]["cv_pct"],
            "cavs_g1": cavs[coluna]["g1"],
            "nets_iqr": nets[coluna]["iqr"],
            "nets_faixa": f"[{nets[coluna]['q1']} – {nets[coluna]['q3']}]",
            "nets_cv": nets[coluna]["cv_pct"],
            "nets_g1": nets[coluna]["g1"],
            "diferenca": f"{dif:.2f}",
            "mais_concentrado": mais_concentrado,
            "mais_espalhado": mais_espalhado,
        }

    return quadro


def graficoIqrVsAmplitudeDesvio(df: pd.DataFrame, time: str):
    estat = calcularIqrTime(df, time)
    nomes_eixo = [estat[c]["nome_exibicao"] for c in colunas]
    valores_iqr = [estat[c]["iqr_num"] for c in colunas]
    valores_dp = [estat[c]["dp_num"] for c in colunas]
    valores_amp = [estat[c]["amplitude_num"] for c in colunas]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            name="Amplitude Total (Máx − Mín)",
            x=nomes_eixo,
            y=valores_amp,
            marker_color="#94A3B8",
            text=[f"{v:.2f}" for v in valores_amp],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Amplitude Total: %{y:.2f}<extra></extra>"
        )
    )

    fig.add_trace(
        go.Bar(
            name="Intervalo Interquartil (IQR = Q₃ − Q₁)",
            x=nomes_eixo,
            y=valores_iqr,
            marker_color="#4F46E5",
            text=[f"{v:.2f}" for v in valores_iqr],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>IQR (50% central): %{y:.2f}<extra></extra>"
        )
    )

    fig.add_trace(
        go.Bar(
            name="Desvio Padrão Amostral (s)",
            x=nomes_eixo,
            y=valores_dp,
            marker_color="#0D9488",
            text=[f"{v:.2f}" for v in valores_dp],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Desvio Padrão (s): %{y:.2f}<extra></extra>"
        )
    )

    fig.update_layout(
        title=f"Comparação de Robustez: Amplitude Total vs. IQR vs. Desvio Padrão (s) — {time}",
        xaxis_title="Variáveis Analisadas",
        yaxis_title="Valor na Unidade Original da Variável",
        barmode="group",
        template="plotly_white",
        yaxis=dict(range=[0, max(valores_amp) * 1.18]),
        legend_title="Medidas de Dispersão",
        height=490
    )

    return fig


def graficoComparativoIqrTimes(df: pd.DataFrame):
    cavs = calcularIqrTime(df, "Cleveland Cavaliers")
    nets = calcularIqrTime(df, "Brooklyn Nets")

    nomes_eixo = [METADADOS_VARIAVEIS[c]["nome"] for c in colunas]
    valores_cavs = [cavs[c]["iqr_num"] for c in colunas]
    valores_nets = [nets[c]["iqr_num"] for c in colunas]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            name="Cleveland Cavaliers",
            x=nomes_eixo,
            y=valores_cavs,
            marker_color="#860038",
            text=[f"{v:.2f}" for v in valores_cavs],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Cleveland Cavaliers — IQR: %{y:.2f}<extra></extra>"
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
            hovertemplate="<b>%{x}</b><br>Brooklyn Nets — IQR: %{y:.2f}<extra></extra>"
        )
    )

    fig.update_layout(
        title="Comparação da Dispersão Central (IQR = Q₃ − Q₁) — Cleveland Cavaliers vs. Brooklyn Nets",
        xaxis_title="Variáveis Analisadas",
        yaxis_title="Intervalo Interquartil (IQR na unidade da variável)",
        barmode="group",
        template="plotly_white",
        yaxis=dict(range=[0, max(max(valores_cavs), max(valores_nets)) * 1.2]),
        legend_title="Equipes",
        height=480
    )

    return fig
