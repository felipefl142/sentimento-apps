"""Dashboard Streamlit: sentimento, alertas, termos/topicos, reviews, modelos, monitoramento.

    streamlit run dashboard/app.py
"""
from __future__ import annotations

import json
import sys
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

from config import settings  # noqa: E402
from dashboard import style  # noqa: E402

st.set_page_config(page_title="Sentimento de apps", page_icon="💬", layout="wide")

try:
    MODE = "dark" if st.context.theme.type == "dark" else "light"
except AttributeError:
    MODE = "light"
SERIES, INK, SENT = style.SERIES[MODE], style.INK[MODE], style.SENTIMENT[MODE]
MODEL_COLOR = {k: SERIES[i] for k, i in style.ENTITY_SLOT.items()}
APPS = settings.load_apps()["apps"]
APP_NAME = {k: v["nome"] for k, v in APPS.items()}
LABELS = ["neg", "neu", "pos"]


# ---------------------------------------------------------------- dados (cache por mtime)
def _mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_data(show_spinner=False)
def _read_parquet(path: str, _mtime: float) -> pd.DataFrame | None:
    p = Path(path)
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    for c in ("data", "semana", "data_ref"):
        if c in df:
            df[c] = pd.to_datetime(df[c])
    return df


def parquet(layer: str, name: str) -> pd.DataFrame | None:
    path = settings.DATA_DIR / layer / f"{name}.parquet"
    return _read_parquet(str(path), _mtime(path))


def report(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


@st.cache_data(ttl=60, show_spinner=False)
def model_versions() -> pd.DataFrame:
    from src.modeling import registry
    client = registry.setup()
    # search_model_versions nao preenche aliases -> champion via get_model_version_by_alias
    champ = registry.champion_version(client)
    rows = [{"versão": int(mv.version),
             "champion": champ is not None and str(mv.version) == str(champ.version),
             "modelo": style.ENTITY_LABEL.get(mv.tags.get("kind"), mv.tags.get("kind")),
             "macro-F1 teste": float(mv.tags.get("primary_value", "nan")),
             "dados até": mv.tags.get("data_max_date"), "promoção": mv.tags.get("promotion"),
             "run_id": mv.run_id}
            for mv in client.search_model_versions(f"name='{registry.MODEL_NAME}'")]
    return pd.DataFrame(rows).sort_values("versão", ascending=False) if rows else pd.DataFrame()


@st.cache_resource(show_spinner="Carregando o champion…")
def champion_model(version: str):
    from src.modeling import registry
    model, _ = registry.load_champion(registry.setup())
    return model


@st.cache_data(ttl=300, show_spinner=False)
def query_reviews(app: str, start, end, stars: tuple, preds: tuple, search: str,
                  only_disagree: bool, _mtime: float, limit: int = 1000) -> tuple[pd.DataFrame, dict]:
    from src.utils import db
    where = ["r.app = $app", "r.data BETWEEN $start AND $end", "r.n_chars > 0"]
    params = {"app": app, "start": start, "end": end}
    if stars:
        where.append(f"r.label IN ({', '.join(repr(s) for s in stars)})")
    if preds:
        where.append(f"s.pred IN ({', '.join(repr(p) for p in preds)})")
    if search:
        where.append("r.text ILIKE $q")
        params["q"] = f"%{search}%"
    if only_disagree:
        where.append("s.pred <> r.label")
    base = f"""FROM silver.reviews r LEFT JOIN gold.review_scores s USING (app, review_id)
               WHERE {' AND '.join(where)}"""
    stats = db.query(f"SELECT count(*) AS n, avg((s.pred <> r.label)::INT) AS disc {base}",
                     **params).iloc[0].to_dict()
    df = db.query(f"""SELECT r.at_local AS quando, r.score AS estrelas, r.label AS rotulo,
                             s.pred AS modelo, s.p_neg, s.p_neu, s.p_pos, r.thumbs_up AS util,
                             r.app_version AS versao, r.text AS texto
                      {base} ORDER BY r.at_utc DESC LIMIT {int(limit)}""", **params)
    return df, stats


def status_badge(kind: str, text: str) -> None:
    fn = {"ok": st.success, "aviso": st.warning, "critico": st.error}[kind]
    fn(f"{style.STATUS_LABEL[kind]} — {text}")


def pct(v) -> str:
    return "—" if v is None or pd.isna(v) else f"{v:.1%}".replace(".", ",")


def weekly(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Media semanal ponderada pelo volume diario (semana comecando na segunda)."""
    d = df.assign(semana=df["data"].dt.to_period("W-SUN").dt.start_time)
    agg = d.groupby(["app", "semana"]).apply(
        lambda g: pd.Series({**{c: np.average(g[c], weights=g["n"]) for c in cols},
                             "n": g["n"].sum(), "dias": len(g)}), include_groups=False)
    return agg.reset_index()


# ---------------------------------------------------------------- cabecalho
daily = parquet("gold", "sentimento_diario")
train_sum = report(settings.REPORTS_DIR / "train_summary.json")
mon = report(settings.REPORTS_DIR / "monitoring.json")

st.title("Sentimento de reviews · Google Play")
if daily is None or mon is None:
    st.info("Sem dados ainda. Rode `make ingest-full && make all` e recarregue.")
    st.stop()

champ = mon["champion"]
st.caption(
    f"{len(APPS)} apps · reviews pt-BR de {daily['data'].min():%d/%m/%Y} a "
    f"{daily['data'].max():%d/%m/%Y} · rótulo pelas estrelas (1–2★ negativo, 3★ neutro, "
    f"4–5★ positivo) · champion v{champ['version']} "
    f"({style.ENTITY_LABEL.get(champ['kind'], champ['kind'])})")

app = st.selectbox("App em foco", list(APPS), format_func=lambda a: APP_NAME[a],
                   help="Usado nas abas Alertas, Termos e tópicos e Explorar reviews")

tab_ov, tab_al, tab_te, tab_rv, tab_mo, tab_mon, tab_try = st.tabs(
    ["Visão geral", "Alertas", "Termos e tópicos", "Explorar reviews", "Modelos",
     "Monitoramento", "Teste um texto"])

# ---------------------------------------------------------------- visao geral
with tab_ov:
    last = daily["data"].max()
    cur = daily[daily["data"] > last - timedelta(days=7)]
    prev = daily[(daily["data"] <= last - timedelta(days=7))
                 & (daily["data"] > last - timedelta(days=14))]

    def wavg(df, col):
        return float(np.average(df[col], weights=df["n"])) if len(df) else np.nan

    st.markdown("**% de reviews negativas (modelo) nos últimos 7 dias** · variação vs 7 dias "
                "anteriores")
    cols = st.columns(len(APPS))
    for col, a in zip(cols, APPS, strict=True):
        c, p = cur[cur["app"] == a], prev[prev["app"] == a]
        v, pv = wavg(c, "pct_neg_modelo"), wavg(p, "pct_neg_modelo")
        col.metric(APP_NAME[a], pct(v),
                   delta=None if np.isnan(pv) else f"{(v - pv) * 100:+.1f}".replace(".", ",") + " p.p.",
                   delta_color="inverse",
                   help=f"{int(c['n'].sum()):,} reviews · {wavg(c, 'estrelas_media'):.2f}★ "
                        "médio".replace(",", "."))

    wk = weekly(daily, ["net_modelo", "net_estrelas", "pct_neg_modelo", "pct_neu_modelo",
                        "pct_pos_modelo"])
    wk = wk[wk["dias"] >= 4]  # semanas incompletas (borda do backfill / hoje) enganam
    apps = list(APPS)
    fig = make_subplots(rows=len(apps), cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        subplot_titles=[APP_NAME[a] for a in apps])
    for i, a in enumerate(apps, start=1):
        g = wk[wk["app"] == a]
        fig.add_trace(go.Scatter(x=g["semana"], y=g["net_modelo"], name="Modelo", mode="lines",
                                 line=dict(color=INK["primary"], width=2), legendgroup="m",
                                 showlegend=i == 1, hovertemplate="%{y:.1%}"), row=i, col=1)
        fig.add_trace(go.Scatter(x=g["semana"], y=g["net_estrelas"], name="Estrelas", mode="lines",
                                 line=dict(color=INK["muted"], width=2, dash="dot"),
                                 legendgroup="e", showlegend=i == 1,
                                 hovertemplate="%{y:.1%}"), row=i, col=1)
    fig.update_layout(title="Sentimento líquido semanal (% positivas − % negativas)")
    style.layout(fig, MODE, height=150 * len(apps) + 100)
    fig.update_yaxes(tickformat=".0%")
    fig.update_annotations(font=dict(size=12, color=INK["secondary"]), x=0, xanchor="left")
    st.plotly_chart(fig, width="stretch")
    st.caption("Cada app em seu painel (escalas e períodos diferentes: o backfill vai até 365 "
               "dias ou 80 mil reviews por app). A distância entre as linhas é o viés do modelo "
               "em relação às estrelas.")

    g = wk[wk["app"] == app]
    fig = go.Figure()
    for lab in LABELS:
        fig.add_trace(go.Bar(x=g["semana"], y=g[f"pct_{lab}_modelo"],
                             name=style.SENT_LABEL[lab], marker_color=SENT[lab],
                             marker_line=dict(color=INK["surface"], width=2),
                             hovertemplate=f"{style.SENT_LABEL[lab]}: %{{y:.1%}}<extra></extra>"))
    fig.update_layout(barmode="stack", title=f"{APP_NAME[app]} · composição semanal (modelo)")
    style.layout(fig, MODE, height=340)
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------- alertas
with tab_al:
    ad = parquet("gold", "alertas_diario")
    al = parquet("gold", "alertas")
    if ad is None:
        st.info("Rode `make analyze`.")
    else:
        cfg_al = settings.load_model_cfg()["alerts"]
        g = ad[ad["app"] == app].sort_values("data")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=g["data"], y=g["baseline_neg"], name="Linha de base (mediana 28d)",
                                 line=dict(color=INK["muted"], width=2, dash="dot"),
                                 hovertemplate="base %{y:.1%}<extra></extra>"))
        fig.add_trace(go.Scatter(x=g["data"], y=g["pct_neg_modelo"], name="% negativas (modelo)",
                                 line=dict(color=INK["primary"], width=1.5),
                                 customdata=g["n"],
                                 hovertemplate="%{y:.1%} · %{customdata} reviews<extra></extra>"))
        hits = al[al["app"] == app] if al is not None and len(al) else pd.DataFrame()
        if len(hits):
            fig.add_trace(go.Scatter(
                x=hits["data"], y=hits["pct_neg"], mode="markers", name="⚠ Alerta",
                marker=dict(color=style.STATUS["critico"], size=11, symbol="triangle-up",
                            line=dict(color=INK["surface"], width=2)),
                customdata=np.stack([hits["tipo"], hits["termos"].str.slice(0, 80)], axis=-1),
                hovertemplate="⚠ %{customdata[0]}<br>%{customdata[1]}<extra></extra>"))
        fig.update_layout(title=f"{APP_NAME[app]} · % diário de reviews negativas")
        style.layout(fig, MODE, height=380, hovermode="closest")
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, width="stretch")
        st.caption(f"Alerta de negatividade: z robusto ≥ {cfg_al['z']} contra os "
                   f"{cfg_al['window']} dias anteriores (mediana/MAD, com piso binomial), "
                   f"alta ≥ {cfg_al['min_delta']:.0%} e ≥ {cfg_al['min_n']} reviews no dia. "
                   f"Alerta de volume: z ≥ {cfg_al['volume_z']} de log(reviews).")
        if not len(hits):
            status_badge("ok", f"nenhum alerta para {APP_NAME[app]} no período")
        else:
            st.markdown(f"**{len(hits)} alertas** · mais recentes primeiro")
            for r in hits.sort_values("data", ascending=False).head(15).itertuples():
                ver = (f" · versão dominante **{r.versao_dominante}** ({r.versao_share:.0%} das "
                       "negativas)") if r.versao_dominante else ""
                with st.expander(f"⚠ {r.data:%d/%m/%Y} · {r.tipo} · {pct(r.pct_neg)} negativas "
                                 f"(base {pct(r.baseline_neg)}) · {r.n} reviews"):
                    st.markdown(f"Termos que dispararam nas negativas do dia: **{r.termos or '—'}**"
                                f"{ver}")
                    for ex in r.exemplos:
                        st.markdown(f"> {ex}")

# ---------------------------------------------------------------- termos e topicos
with tab_te:
    ts = parquet("gold", "termos_sentimento")
    te = parquet("gold", "termos_emergentes")
    tp = parquet("gold", "topicos")
    if ts is None:
        st.info("Rode `make analyze`.")
    else:
        t_app = ts[ts["app"] == app]
        periods = sorted(t_app["periodo"].unique(), key=lambda p: (p != "90d", p), reverse=False)
        periods = ["90d"] + sorted([p for p in periods if p != "90d"], reverse=True)
        per = st.selectbox("Período", periods,
                           format_func=lambda p: "Últimos 90 dias" if p == "90d" else p)
        c1, c2 = st.columns(2)
        for col, lado, title in ((c1, "neg", "Distinguem as negativas"),
                                 (c2, "pos", "Distinguem as positivas")):
            g = t_app[(t_app["periodo"] == per) & (t_app["lado"] == lado)].head(15) \
                .sort_values("z")
            fig = go.Figure(go.Bar(x=g["z"], y=g["termo"], orientation="h",
                                   marker_color=SENT[lado], customdata=g["n"],
                                   hovertemplate="%{y}: z %{x:.1f} · %{customdata}×<extra></extra>"))
            fig.update_layout(title=title)
            style.layout(fig, MODE, height=460, legend=False, hovermode="closest")
            fig.update_xaxes(title="z (log-odds, prior de Dirichlet)")
            col.plotly_chart(fig, width="stretch")
        st.caption("Estrelas como rótulo; negativas × positivas do mesmo período e app. "
                   "Termos sem acento, sem stopwords, com bigramas. Escalas independentes: o z "
                   "cresce com o volume, e positivas são ~4× mais numerosas.")

        st.subheader("Termos emergentes nas reclamações")
        e = te[te["app"] == app] if te is not None else pd.DataFrame()
        if len(e):
            r0 = e.iloc[0]
            st.caption(f"Negativas dos últimos {int(r0['janela'])} dias até "
                       f"{r0['data_ref']:%d/%m} ({int(r0['n_alvo'])} reviews) × 90 dias "
                       f"anteriores ({int(r0['n_base'])})")
            g = e.head(15).sort_values("z")
            fig = go.Figure(go.Bar(x=g["z"], y=g["termo"], orientation="h",
                                   marker_color=SENT["neg"], customdata=g["n"],
                                   hovertemplate="%{y}: z %{x:.1f} · %{customdata}×<extra></extra>"))
            style.layout(fig, MODE, height=420, legend=False, hovermode="closest")
            fig.update_xaxes(title="z (recente × referência)")
            st.plotly_chart(fig, width="stretch")
        else:
            cfg_t = settings.load_model_cfg()["terms"]
            status_badge("ok", f"nenhum termo cresceu de forma relevante (z ≥ {cfg_t['min_z']}) "
                         f"nas negativas dos últimos {cfg_t['recent_days']} dias × "
                         f"{cfg_t['background_days']} dias anteriores")

        st.subheader("Tópicos das reclamações recentes (NMF)")
        t = tp[tp["app"] == app] if tp is not None and len(tp) else pd.DataFrame()
        if not len(t):
            st.info("Reviews negativas insuficientes para tópicos.")
        else:
            st.caption(f"Negativas de {t['data_min'].iloc[0]:%d/%m} a {t['data_max'].iloc[0]:%d/%m}"
                       " · participação = fração das reviews cujo tópico dominante é este")
            for r in t.itertuples():
                with st.expander(f"{r.share:.0%} · {r.rotulo}  ({r.n_docs} reviews)"):
                    st.markdown(f"**Termos:** {r.termos}")
                    for ex in r.exemplos:
                        st.markdown(f"> {ex}")

# ---------------------------------------------------------------- explorar reviews
with tab_rv:
    dmin, dmax = daily[daily["app"] == app]["data"].agg(["min", "max"])
    f1, f2, f3, f4 = st.columns([2, 1.2, 1.2, 2])
    rng = f1.date_input("Período", (dmax.date() - timedelta(days=6), dmax.date()),
                        min_value=dmin.date(), max_value=dmax.date())
    stars = f2.multiselect("Estrelas", LABELS, format_func=style.SENT_LABEL.get,
                           placeholder="Todas")
    preds = f3.multiselect("Modelo", LABELS, format_func=style.SENT_LABEL.get,
                           placeholder="Todas")
    search = f4.text_input("Buscar no texto", placeholder="ex.: pix, entrega, login")
    only_dis = st.checkbox("Só discordâncias (modelo ≠ estrelas)",
                           help="Boa parte é ruído de rótulo: texto negativo com 5★, elogio com 1★…")
    if isinstance(rng, tuple) and len(rng) == 2:
        path = settings.GOLD_DIR / "review_scores.parquet"
        df, stats = query_reviews(app, str(rng[0]), str(rng[1]), tuple(stars), tuple(preds),
                                  search.strip(), only_dis, _mtime(path))
        st.caption(f"{int(stats['n']):,} reviews".replace(",", ".")
                   + (f" · modelo discorda das estrelas em {pct(stats['disc'])}"
                      if stats["n"] else "") + " · mostrando as 1.000 mais recentes")
        df["rotulo"] = df["rotulo"].map(style.SENT_LABEL)
        df["modelo"] = df["modelo"].map(style.SENT_LABEL)
        prob = {f"p_{c}": st.column_config.ProgressColumn(f"P({c})", min_value=0, max_value=1,
                                                           format="%.2f", color=SENT[c])
                for c in LABELS}
        st.dataframe(df, width="stretch", hide_index=True, height=520, column_config={
            "quando": st.column_config.DatetimeColumn("Quando", format="DD/MM HH:mm"),
            "estrelas": st.column_config.NumberColumn("★", format="%d"),
            "rotulo": "Estrelas →", "modelo": "Modelo", "util": "👍", "versao": "Versão",
            "texto": st.column_config.TextColumn("Texto", width="large"), **prob})

# ---------------------------------------------------------------- modelos
with tab_mo:
    if train_sum is None:
        st.info("Rode `make train`.")
    else:
        sp = train_sum["split"]
        st.caption(f"Split temporal: treino até {sp['valid_start']} (excl.) · valid "
                   f"{sp['valid_start']} → {sp['test_start']} · teste {sp['test_start']} → "
                   f"{sp['data_max']} ({sp['n_test']:,} reviews, distribuição real)"
                   .replace(",", "."))
        cands = train_sum["candidates"]
        rows = [{"modelo": style.ENTITY_LABEL.get(k, k), "macro-F1": m["macro_f1"],
                 "acurácia": m["accuracy"], "F1 neg": m["f1_neg"], "F1 neu": m["f1_neu"],
                 "F1 pos": m["f1_pos"], "vencedor": "🏆" if k == train_sum["best"] else ""}
                for k, m in cands.items()]
        st.dataframe(pd.DataFrame(rows).sort_values("macro-F1", ascending=False),
                     hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.3f")
                                    for c in ("macro-F1", "acurácia", "F1 neg", "F1 neu", "F1 pos")})
        verdict, v = train_sum["reason"], train_sum["registered_version"]
        if train_sum["promoted"]:
            status_badge("ok", f"v{v} promovida a champion · {verdict}")
        else:
            status_badge("aviso", f"v{v} rejeitada pelo gate · {verdict}")

        fig = go.Figure()
        for k, m in cands.items():
            fig.add_trace(go.Bar(x=[APP_NAME.get(a, a) for a in APPS],
                                 y=[m.get(f"macro_f1__{a}") for a in APPS],
                                 name=style.ENTITY_LABEL.get(k, k),
                                 marker_color=MODEL_COLOR.get(k, INK["muted"]),
                                 marker_line=dict(color=INK["surface"], width=2),
                                 hovertemplate="%{x}: %{y:.3f}<extra>" +
                                 style.ENTITY_LABEL.get(k, k) + "</extra>"))
        fig.update_layout(barmode="group", title="Macro-F1 no teste por app")
        style.layout(fig, MODE, height=360, hovermode="closest")
        fig.update_yaxes(range=[0, 1])
        st.plotly_chart(fig, width="stretch")

        bt = parquet("gold", "backtest_predictions")
        if bt is not None:
            kinds = [k for k in cands if k in set(bt["kind"])]
            k = st.radio("Matriz de confusão", kinds, horizontal=True,
                         format_func=lambda x: style.ENTITY_LABEL.get(x, x),
                         index=kinds.index(train_sum["best"]) if train_sum["best"] in kinds else 0)
            g = bt[bt["kind"] == k]
            cm = pd.crosstab(g["label"], g["pred"]).reindex(index=LABELS, columns=LABELS,
                                                            fill_value=0)
            rown = cm.div(cm.sum(1), axis=0)
            txt = [[f"{rown.iloc[i, j]:.0%}<br>({cm.iloc[i, j]:,})".replace(",", ".")
                    for j in range(3)] for i in range(3)]
            fig = go.Figure(go.Heatmap(
                z=rown.values, x=[f"previsto {style.SENT_LABEL[c]}" for c in LABELS],
                y=[f"real {style.SENT_LABEL[c]}" for c in LABELS], text=txt,
                texttemplate="%{text}",
                colorscale=[[i / 6, c] for i, c in enumerate(style.SEQ_BLUE[MODE])],
                zmin=0, zmax=1, showscale=False, xgap=2, ygap=2,
                hovertemplate="%{y} → %{x}: %{z:.1%}<extra></extra>"))
            fig.update_layout(title=f"{style.ENTITY_LABEL.get(k, k)} · normalizada por linha "
                              "(recall de cada classe)")
            style.layout(fig, MODE, height=380, legend=False, hovermode="closest")
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, width="stretch")

        with st.expander("Versões no MLflow Model Registry"):
            try:
                mv = model_versions()
                st.dataframe(mv, hide_index=True, width="stretch")
            except Exception as e:  # noqa: BLE001
                st.warning(f"MLflow indisponível: {e}")
            st.caption("UI completa: `make mlflow` → http://localhost:5000")

# ---------------------------------------------------------------- monitoramento
with tab_mon:
    fr, perf = mon["freshness"], mon["performance"]
    if fr["retrain_recommended"]:
        status_badge("aviso", "retreino recomendado: " + "; ".join(fr["reasons"]))
    else:
        status_badge("ok", "sem sinais de degradação — retreino não necessário")
    if fr["stale_apps"]:
        status_badge("aviso", "sem reviews novas há dias: "
                     + ", ".join(APP_NAME.get(a, a) for a in fr["stale_apps"]))
    c = st.columns(4)
    c[0].metric("Macro-F1 no teste", f"{perf['test_macro_f1']:.3f}")
    c[1].metric("Macro-F1 ao vivo", "—" if perf["live_macro_f1"] is None
                else f"{perf['live_macro_f1']:.3f}",
                delta=None if perf["drop"] is None else f"{-perf['drop']:+.3f}",
                help=f"Reviews a partir de {perf['live_from']} ({perf['live_n']:,} com rótulo)"
                .replace(",", "."))
    age = fr["model_data_age_days"]
    c[2].metric("Idade dos dados do champion", f"{age} dia" + ("s" if age != 1 else ""))
    c[3].metric("Última coleta", (fr["last_ingest"] or "—")[:16].replace("T", " "),
                help="UTC")

    ls = parquet("monitoring", "live_scores")
    if ls is not None and len(ls):
        g = ls[ls["app"] == "todos"]
        fig = go.Figure()
        fig.add_hline(y=perf["test_macro_f1"], line=dict(color=INK["muted"], dash="dot", width=2),
                      annotation_text="teste", annotation_font_color=INK["secondary"])
        fig.add_trace(go.Scatter(x=g["semana"], y=g["macro_f1"], mode="lines+markers",
                                 name="Ao vivo (todos os apps)", customdata=g["n"],
                                 line=dict(color=MODEL_COLOR.get(champ["kind"], INK["primary"]),
                                           width=2), marker=dict(size=8),
                                 hovertemplate="%{y:.3f} · %{customdata} reviews<extra></extra>"))
        fig.update_layout(title="Macro-F1 semanal do champion em dados posteriores ao treino")
        style.layout(fig, MODE, height=320)
        st.plotly_chart(fig, width="stretch")
    else:
        st.caption("Performance ao vivo aparece quando chegarem reviews posteriores aos dados "
                   "de treino do champion (a estrela é o rótulo, então é imediato após `make update`).")

    rows = []
    for a, d in mon["drift"].items():
        rows.append({"app": APP_NAME.get(a, a), "reviews recentes": d["n_recent"],
                     "PSI rótulo": d["label_psi"], "% neg ref": d["label_dist_ref"]["neg"],
                     "% neg recente": d["label_dist_recent"]["neg"],
                     "JS vocabulário": d["js"], "JS no teste": d["js_test"],
                     "OOV": d["oov_rate"], "OOV no teste": d["oov_rate_test"],
                     "modelo × estrelas (TV)": d["pred_vs_stars_tv"]})
    st.markdown(f"**Drift** · recente = {mon['window']['recent_start']} → {mon['window']['last']}"
                f" · referência = {mon['window']['ref']}")
    fmt3 = st.column_config.NumberColumn(format="%.3f")
    fmtp = st.column_config.NumberColumn(format="percent")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
        "PSI rótulo": fmt3, "JS vocabulário": fmt3, "JS no teste": fmt3,
        "modelo × estrelas (TV)": fmt3, "% neg ref": fmtp, "% neg recente": fmtp,
        "OOV": fmtp, "OOV no teste": fmtp})
    st.caption("PSI > 0,2 no rótulo ou JS do vocabulário muito acima do observado no teste "
               "recomendam retreino. TV = variação total entre a distribuição prevista e a das "
               "estrelas.")
    q = report(settings.GOLD_DIR / "quality_checks.json")
    if q:
        with st.expander(("✅" if q["ok"] else "❌") + " Checagens de qualidade do silver"):
            st.dataframe(pd.DataFrame(q["checks"]), hide_index=True, width="stretch")

# ---------------------------------------------------------------- teste um texto
with tab_try:
    txt = st.text_area("Escreva uma review", "O app é bom, mas desde a última atualização o "
                       "pix não funciona e o suporte não responde.", height=100)
    if st.button("Classificar", type="primary") and txt.strip():
        model = champion_model(champ["version"])
        out = model.predict(pd.DataFrame({"text": [txt]})).iloc[0]
        st.markdown(f"**{style.SENT_LABEL[out['pred']]}** · champion v{champ['version']} "
                    f"({style.ENTITY_LABEL.get(champ['kind'], champ['kind'])})")
        fig = go.Figure(go.Bar(
            x=[out[f"p_{c}"] for c in LABELS], y=[style.SENT_LABEL[c] for c in LABELS],
            orientation="h", marker_color=[SENT[c] for c in LABELS],
            text=[pct(out[f"p_{c}"]) for c in LABELS], textposition="outside",
            textfont=dict(color=INK["secondary"]),
            hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
        style.layout(fig, MODE, height=220, legend=False, hovermode="closest")
        fig.update_xaxes(range=[0, 1.1], tickformat=".0%")
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, width="stretch")
