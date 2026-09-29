"""Tokens visuais do dashboard (paleta de referencia validada p/ CVD, claro e escuro).

- Sentimento e DIVERGENTE: neg vermelho <-> pos azul, neutro cinza (o meio le como "nada").
  O validador acusa o par cinza x vermelho no escuro (protan 6.4) -> sempre com legenda,
  hover rotulado e 2px de respiro entre segmentos (codificacao secundaria).
- Modelos (entidades): slots categoricos 1-3 em ordem fixa (validados all-pairs). Cor segue
  o modelo em todos os graficos. Apps NUNCA sao distinguidos por cor (5 > 3 slots): small
  multiples ou seletor.
- Modelo x estrelas numa mesma serie temporal: tinta primaria (modelo) x secundaria
  tracejada (estrelas) - nao compete com as cores de sentimento.
- Status (ok / aviso / critico): paleta reservada, sempre com icone + rotulo.
"""
from __future__ import annotations

import plotly.graph_objects as go

SERIES = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a"],
    "dark": ["#3987e5", "#d95926", "#199e70"],
}
SENTIMENT = {
    "light": {"neg": "#e34948", "neu": "#a8a79f", "pos": "#2a78d6"},
    "dark": {"neg": "#e66767", "neu": "#6f6e68", "pos": "#3987e5"},
}
SENT_LABEL = {"neg": "Negativo", "neu": "Neutro", "pos": "Positivo"}
# heatmap (matriz de confusao): rampa sequencial azul; o valor baixo fica perto da superficie
# (claro no modo claro, escuro no modo escuro) - passos proprios por modo, nao inversao automatica
SEQ_BLUE = {
    "light": ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
    "dark": ["#0d366b", "#104281", "#184f95", "#1c5cab", "#256abf", "#3987e5", "#6da7ec"],
}
INK = {
    "light": {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781",
              "grid": "#e1e0d9", "axis": "#c3c2b7", "surface": "#fcfcfb", "wash": "#f0efec"},
    "dark": {"primary": "#ffffff", "secondary": "#c3c2b7", "muted": "#898781",
             "grid": "#2c2c2a", "axis": "#383835", "surface": "#1a1a19", "wash": "#383835"},
}

ENTITY_SLOT = {"bertimbau_ft": 0, "xlmr_zeroshot": 1, "tfidf_lr": 2}
ENTITY_LABEL = {"bertimbau_ft": "BERTimbau fine-tune", "xlmr_zeroshot": "XLM-R zero-shot",
                "tfidf_lr": "TF-IDF + LogReg"}

STATUS = {"ok": "#0ca30c", "aviso": "#fab219", "critico": "#d03b3b"}
STATUS_LABEL = {"ok": "🟢 OK", "aviso": "🟡 Atenção", "critico": "🔴 Crítico"}


def rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def layout(fig: go.Figure, mode: str, height: int = 360, legend: bool = True,
           hovermode: str = "x unified") -> go.Figure:
    ink = INK[mode]
    axis = dict(gridcolor=ink["grid"], linecolor=ink["axis"], zeroline=False, automargin=True,
                tickfont=dict(color=ink["muted"]), title_font=dict(color=ink["secondary"]))
    fig.update_layout(
        # margens fixas: automargin mede o texto antes da fonte carregar e corta os ticks
        height=height, margin=dict(l=72, r=16, t=40, b=52),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif",
                  color=ink["primary"], size=13),
        title_font=dict(size=14, color=ink["primary"]),
        hovermode=hovermode, showlegend=legend, bargap=0.15,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                    font=dict(color=ink["secondary"])),
        hoverlabel=dict(bgcolor=ink["surface"], font_color=ink["primary"],
                        bordercolor=ink["axis"]),
    )
    if fig.layout.title.text:  # titulo no topo do container, legenda logo abaixo dele
        fig.update_layout(margin=dict(t=84), title=dict(x=0, xanchor="left", y=0.98,
                                                        yref="container", yanchor="top"))
    fig.update_xaxes(**axis, showgrid=False)
    fig.update_yaxes(**axis)
    return fig
