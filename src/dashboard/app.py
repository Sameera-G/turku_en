from __future__ import annotations

# Make `src.*` imports work even when Streamlit launches this file directly.
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import LIVE_SOURCE_HEALTH_JSON, ONLINE_MARKET_STATE_JSON, LIVE_LONG_PARQUET
from src.online.store import OnlineStore
from src.online.adaptive import AdaptiveState
from src.online.continuous import default_model_path
from src.models_v8 import load_any_bundle
from src.explainability import grouped_model_importance, local_lgb_group_contributions, LIVE_SIGNAL_GROUPS, select_live_group
from src.features import build_features
from src.performance import (
    official_validation_summary, validation_detail, validation_step_performance,
    validation_intuitive_accuracy, live_summary, live_step_performance,
    daily_history, latest_daily_report, public_baseline_comparison,
)

ONLINE_FORECAST = ROOT / "outputs" / "online" / "latest_forecast.csv"
ONLINE_DIAG = ROOT / "outputs" / "online" / "latest_diagnostics.csv"
ONLINE_HISTORY = ROOT / "outputs" / "online" / "history"
V10_FORECAST = ROOT / "outputs" / "v10" / "forecast_backtest_v10.csv"
V10_DIAG = ROOT / "outputs" / "v10" / "forecast_diagnostics_v10.csv"
V8_FORECAST = ROOT / "outputs" / "v8" / "forecast_backtest_v8.csv"
V8_DIAG = ROOT / "outputs" / "v8" / "forecast_diagnostics_v8.csv"
V3_FORECAST = ROOT / "outputs" / "v3" / "forecast_backtest_v3.csv"
V3_DIAG = ROOT / "outputs" / "v3" / "forecast_diagnostics_v3.csv"
V2_FORECAST = ROOT / "outputs" / "v2" / "forecast_backtest_v2.csv"
V1_FORECAST = ROOT / "outputs" / "forecast_backtest.csv"
HELSINKI = ZoneInfo("Europe/Helsinki")

st.set_page_config(page_title="TE-BalanceAI V11", layout="wide", page_icon="⚡")
st.title("Turku Energia – Imbalance Forecast Cockpit V11")
st.caption(
    "Live hourly forecasting • V11 adaptive tournament + V10 full-refit champion + MCP/revision features + sample-aware reliability + asymmetric conformal calibration • 40 × 15-minute steps • next 10 hours • official validation evidence • "
    "24h/7d/14d self-learning • daily champion/challenger checkpoint"
)


def _active_paths() -> tuple[Path | None, Path | None]:
    forecast = next((p for p in [ONLINE_FORECAST, V10_FORECAST, V8_FORECAST, V3_FORECAST, V2_FORECAST, V1_FORECAST] if p.exists()), None)
    if forecast == ONLINE_FORECAST and ONLINE_DIAG.exists():
        return forecast, ONLINE_DIAG
    if forecast == V10_FORECAST and V10_DIAG.exists():
        return forecast, V10_DIAG
    if forecast == V8_FORECAST and V8_DIAG.exists():
        return forecast, V8_DIAG
    if forecast == V3_FORECAST and V3_DIAG.exists():
        return forecast, V3_DIAG
    return forecast, None


def _load_forecast(path: Path, diag_path: Path | None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if diag_path and diag_path.exists():
        diag = pd.read_csv(diag_path)
        df = df.merge(diag, on=["forecast_origin_utc", "delivery_start_utc", "step"], how="left")
    return df


def _previous_online_forecast(current_origin: str) -> pd.DataFrame:
    if not ONLINE_HISTORY.exists():
        return pd.DataFrame()
    current = pd.Timestamp(current_origin)
    files = sorted(ONLINE_HISTORY.glob("forecast_*.csv"))
    candidates = []
    for p in files:
        try:
            d = pd.read_csv(p, nrows=1)
            if d.empty:
                continue
            o = pd.Timestamp(d.iloc[0]["forecast_origin_utc"])
            if o < current:
                candidates.append((o, p))
        except Exception:
            continue
    if not candidates:
        return pd.DataFrame()
    return pd.read_csv(max(candidates, key=lambda z: z[0])[1])


def _fmt(v, digits=2, suffix=""):
    if v is None or pd.isna(v):
        return "n/a"
    return f"{float(v):.{digits}f}{suffix}"


def _status_direction(row: pd.Series) -> tuple[str, float]:
    probs = {"UP": float(row.get("p_up", 0)), "DOWN": float(row.get("p_down", 0)), "NONE": float(row.get("p_none", 0))}
    label = max(probs, key=probs.get)
    return label, probs[label]


forecast_path, diag_path = _active_paths()
if forecast_path is None:
    st.error("No forecast is available yet. Run scripts/run_online_cycle.py or the V3 backtest pipeline first.")
    st.stop()

forecast_all = _load_forecast(forecast_path, diag_path)
st.caption(f"Active forecast source: `{forecast_path.relative_to(ROOT)}`")

live_tab, accuracy_tab, learning_tab, signals_tab, replay_tab, health_tab = st.tabs([
    "⚡ Live forecast", "🎯 Accuracy & validation", "🧠 Self-learning", "🔎 Signals & explainability", "🕘 Historical replay", "🛰 Data health"
])

with live_tab:
    origins = forecast_all["forecast_origin_utc"].drop_duplicates().tolist()
    default_origin = origins[-1]
    origin = st.selectbox("Forecast origin", origins, index=len(origins)-1, key="live_origin")
    x = forecast_all[forecast_all["forecast_origin_utc"] == origin].copy()
    x["delivery_start_utc"] = pd.to_datetime(x["delivery_start_utc"], utc=True)
    x["delivery_local"] = x["delivery_start_utc"].dt.tz_convert(HELSINKI)

    previous = _previous_online_forecast(origin) if forecast_path == ONLINE_FORECAST else pd.DataFrame()
    if not previous.empty:
        previous["delivery_start_utc"] = pd.to_datetime(previous["delivery_start_utc"], utc=True)
        pm = previous[["delivery_start_utc", "p50"]].rename(columns={"p50": "previous_p50"})
        x = x.merge(pm, on="delivery_start_utc", how="left")
        x["revision_eur_mwh"] = x["p50"] - x["previous_p50"]
    else:
        x["revision_eur_mwh"] = np.nan

    reliability = x["reliability"].min() if "reliability" in x else np.nan
    source_health = x["source_health"].min() if "source_health" in x else np.nan
    max_revision = x["revision_eur_mwh"].abs().max() if x["revision_eur_mwh"].notna().any() else np.nan
    direction_probs = {
        "UP": float(x["p_up"].max()), "DOWN": float(x["p_down"].max()), "NONE": float(x["p_none"].max())
    }
    dominant = max(direction_probs, key=direction_probs.get)

    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("Max P50", f"{x.p50.max():.1f} €/MWh")
    c2.metric("High-spike risk", f"{100*x.p_spike_high.max():.0f}%")
    c3.metric("Low-spike risk", f"{100*x.p_spike_low.max():.0f}%")
    c4.metric("Dominant direction", f"{dominant} {100*direction_probs[dominant]:.0f}%")
    c5.metric("Forecast steps", f"{len(x)} × 15 min")
    c6.metric("Lowest reliability", "n/a" if pd.isna(reliability) else f"{100*reliability:.0f}%")
    c7.metric("Max hourly revision", "n/a" if pd.isna(max_revision) else f"{max_revision:.1f} €/MWh")

    if "v8_spot_available" in x:
        spot_fraction = pd.to_numeric(x["v8_spot_available"], errors="coerce").fillna(0).mean()
        if spot_fraction >= 0.99:
            st.success("V10 market-rule mode ACTIVE: legal FI day-ahead spot is available, so P50 uses the spot-spread expert ensemble.")
        else:
            st.warning("V10 market-rule mode FALLBACK: live FI spot is missing for some/all delivery periods, so those rows use the validated direct-price model. Configure ENTSO-E/MCP, licensed/manual spot, or the optional public demo fallback.")

    if "conformal_expand_eur" in x:
        expansion = pd.to_numeric(x["conformal_expand_eur"], errors="coerce").fillna(0).max()
        if expansion > 0:
            st.info(f"Live conformal calibration ACTIVE: current intervals receive up to ±{expansion:.1f} €/MWh pooled expansion from settled live errors.")
    if {"reliability_model","reliability_empirical","reliability_source"}.issubset(x.columns):
        with st.expander("How the reliability score is built"):
            comps=[("Trained-model confidence",x["reliability_model"].mean()),
                   ("Empirical live performance",x["reliability_empirical"].mean())]
            if "reliability_calibration" in x:
                comps.append(("Live calibration quality",x["reliability_calibration"].mean()))
            comps.extend([("Source health",x["reliability_source"].mean()),("Final reliability",x["reliability"].mean())])
            rr=pd.DataFrame(comps,columns=["component","score"]); rr["score_%"]=100*rr.pop("score")
            st.dataframe(rr, width="stretch", hide_index=True)
            raw_cov=pd.to_numeric(x["reliability_coverage_raw"],errors="coerce").dropna() if "reliability_coverage_raw" in x else pd.Series(dtype=float)
            shr_cov=pd.to_numeric(x["reliability_coverage_shrunk"],errors="coerce").dropna() if "reliability_coverage_shrunk" in x else pd.Series(dtype=float)
            maturity=pd.to_numeric(x["reliability_live_sample_confidence"],errors="coerce").dropna() if "reliability_live_sample_confidence" in x else pd.Series(dtype=float)
            if len(raw_cov):
                st.caption(f"Raw live coverage: {100*raw_cov.mean():.1f}% | evidence-shrunk coverage: {100*shr_cov.mean():.1f}% | live evidence maturity: {100*maturity.mean():.0f}%")
            if "conformal_expand_low_eur" in x and "conformal_expand_high_eur" in x:
                st.caption(f"Current live interval expansion: -{x['conformal_expand_low_eur'].mean():.1f} / +{x['conformal_expand_high_eur'].mean():.1f} €/MWh")
            if "reliability_reason" in x:
                st.caption(str(x["reliability_reason"].iloc[0]))
            st.info("Reliability is a confidence/review score, not forecast accuracy. V10 makes the live part sample-aware so a tiny sample can warn you without overpowering the much larger historical evidence.")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x.delivery_local, y=x.p90, line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x.delivery_local, y=x.p10, fill="tonexty", name="P10–P90", line=dict(width=0)))
    fig.add_trace(go.Scatter(x=x.delivery_local, y=x.p50, name="Fused P50", mode="lines+markers"))
    if "p50_lgb" in x:
        fig.add_trace(go.Scatter(x=x.delivery_local, y=x.p50_lgb, name="LightGBM", mode="lines", line=dict(dash="dot")))
    if "p50_cat" in x:
        fig.add_trace(go.Scatter(x=x.delivery_local, y=x.p50_cat, name="CatBoost", mode="lines", line=dict(dash="dash")))
    if "p50_direct" in x:
        fig.add_trace(go.Scatter(x=x.delivery_local, y=x.p50_direct, name="Direct-price expert", mode="lines", line=dict(dash="dot")))
    if "p50_spread" in x and pd.to_numeric(x["p50_spread"], errors="coerce").notna().any():
        fig.add_trace(go.Scatter(x=x.delivery_local, y=x.p50_spread, name="Spot-spread expert", mode="lines", line=dict(dash="dash")))
    if x["revision_eur_mwh"].notna().any():
        fig.add_trace(go.Scatter(x=x.delivery_local, y=x["previous_p50"], name="Previous hourly P50", mode="lines", line=dict(dash="dashdot")))
    fig.update_layout(
        title="Live next-10-hours forecast – every point is one 15-minute delivery period",
        yaxis_title="EUR/MWh", xaxis_title="Delivery time Europe/Helsinki", hovermode="x unified", height=520,
    )
    st.plotly_chart(fig, width="stretch")

    risk_cols = ["p_spike_high", "p_spike_low", "p_up", "p_down"]
    review = x.copy()
    review["risk"] = review[risk_cols].max(axis=1)
    rel_penalty = 1 - review["reliability"] if "reliability" in review else 0
    rev_term = review["revision_eur_mwh"].abs().fillna(0) / 100.0
    review["review_score"] = review["risk"] + rel_penalty + rev_term
    review = review.sort_values("review_score", ascending=False).head(8)
    st.subheader("What should I check now?")
    review_cols = ["delivery_local", "p50", "p10", "p90", "p_up", "p_down", "p_spike_high", "p_spike_low", "revision_eur_mwh"]
    for c in ["reliability", "model_disagreement", "source_health", "top_drivers"]:
        if c in review:
            review_cols.append(c)
    st.dataframe(review[review_cols], width="stretch", hide_index=True)

    st.subheader("Complete 15-minute forecast table – all 40 delivery periods")
    table = x.copy()
    dirs = table.apply(_status_direction, axis=1)
    table["direction"] = [d[0] for d in dirs]
    table["direction_confidence"] = [d[1] for d in dirs]
    table["interval_width"] = table["p90"] - table["p10"]
    table["max_spike_risk"] = table[["p_spike_high", "p_spike_low"]].max(axis=1)
    cols = ["step", "delivery_local", "p10", "p50", "p90", "interval_width", "direction", "direction_confidence", "p_spike_high", "p_spike_low"]
    for c in ["p50_direct", "p50_spread", "v8_spot_available", "reliability", "revision_eur_mwh", "model_disagreement", "top_drivers"]:
        if c in table:
            cols.append(c)
    st.dataframe(table[cols], width="stretch", hide_index=True, height=720)

with accuracy_tab:
    st.header("Accuracy – official validation and true live performance")
    st.caption("MAE / pinball / Brier are the challenge metrics. The ±€10/20/50 hit rates are additional intuitive diagnostics, not official scoring metrics.")

    summary = official_validation_summary()
    detail = validation_detail()
    intuitive = validation_intuitive_accuracy(detail)
    step_perf = validation_step_performance(detail)

    if summary:
        st.subheader("Locked official validation evidence (7 Sep – 4 Oct 2026)")
        a, b, c, d, e = st.columns(5)
        a.metric("P50 MAE ↓", f"{float(summary['mae_p50']):.2f} €/MWh")
        b.metric("P10/P90 pinball ↓", f"{float(summary['calibration_pinball_mean']):.3f}")
        c.metric("P10–P90 coverage", f"{100*float(summary['coverage_p10_p90']):.1f}%", help="Target is about 80%; this is supporting calibration information.")
        d.metric("Spike Brier ↓", f"{float(summary['brier_spike_mean']):.4f}")
        e.metric("Direction Brier ↓", f"{float(summary['brier_direction']):.4f}")

        baseline = public_baseline_comparison(summary)
        if baseline:
            st.success(
                f"Organizer public-validation baseline MAE: {baseline['mae_p50']:.2f} €/MWh → locked V8/V9 validation evidence: {float(summary['mae_p50']):.2f} €/MWh "
                f"({baseline['mae_p50_improvement_pct']:.1f}% lower error). "
                f"Pinball improvement: {baseline['calibration_pinball_mean_improvement_pct']:.1f}%; "
                f"spike-Brier improvement: {baseline['brier_spike_mean_improvement_pct']:.1f}%; "
                f"direction-Brier improvement: {baseline['brier_direction_improvement_pct']:.1f}%."
            )

        if intuitive:
            a, b, c, d = st.columns(4)
            a.metric("Median absolute error", f"{intuitive['median_abs_error']:.2f} €/MWh")
            b.metric("Within ±€10", f"{100*intuitive['within_10']:.1f}%")
            c.metric("Within ±€20", f"{100*intuitive['within_20']:.1f}%")
            d.metric("Within ±€50", f"{100*intuitive['within_50']:.1f}%")

        if not step_perf.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=step_perf["step"], y=step_perf["mae"], mode="lines+markers", name="MAE"))
            fig.update_layout(title="Validation MAE by every 15-minute forecast horizon", xaxis_title="Step (1 = origin, 40 = +9h45m)", yaxis_title="MAE EUR/MWh", height=380)
            st.plotly_chart(fig, width="stretch")
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=step_perf["step"], y=100*step_perf["coverage"], mode="lines+markers", name="Coverage"))
            fig2.add_hline(y=80, line_dash="dash", annotation_text="80% target")
            fig2.update_layout(title="P10–P90 coverage by every 15-minute horizon", xaxis_title="Step", yaxis_title="Coverage %", height=340)
            st.plotly_chart(fig2, width="stretch")
            with st.expander("40-step validation table"):
                st.dataframe(step_perf, width="stretch", hide_index=True)
    else:
        st.warning("Official validation score files were not found. Run the V8 pipeline (or V3 fallback) first.")

    refit_report=ROOT / "reports" / "v10" / "production_refit_summary.json"
    if refit_report.exists():
        try:
            rr=json.loads(refit_report.read_text(encoding="utf-8")); ch=rr.get("chronological_refit_experiment",{})
            st.subheader("Production-model chronological holdout evidence")
            full=rr.get("chronological_full_ensemble_experiment",{})
            a,b,c,d=st.columns(4)
            a.metric("Recent legal rows added", f"{int(rr.get('added_recent_legal_rows',0)):,}")
            b.metric("Old fair V8 MAE", f"{float(full.get('old_v8_fair_mae_eur_mwh', ch.get('old_direct_mae_eur_mwh', float('nan')))):.2f} €/MWh")
            c.metric("V10/V11 production holdout MAE", f"{float(full.get('v10_robust_50_50_mae_eur_mwh', ch.get('refit_direct_mae_eur_mwh', float('nan')))):.2f} €/MWh", delta=f"-{float(full.get('relative_mae_reduction_pct', ch.get('relative_mae_reduction_pct',0))):.1f}% error")
            d.metric("Locked organizer MAE", f"{float(rr.get('locked_official_validation_mae_eur_mwh', summary.get('mae_p50', float('nan')) if summary else float('nan'))):.2f} €/MWh")
            st.info("The ~30.8 €/MWh number is a strict later chronological holdout for the production refit, not a replacement for the frozen organizer public-validation score. V11 keeps both visible on purpose: the organizer evidence stays fixed, while production/live evidence can improve as newer legal data are learned.")
        except Exception:
            pass

    st.divider()
    st.subheader("Live settled accuracy")
    live_hours = st.radio("Live performance window", [24, 168, 336], horizontal=True, format_func=lambda h: {24:"24 hours",168:"7 days",336:"14 days"}[h])
    try:
        recent = OnlineStore().recent_joined(pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=live_hours))
    except Exception:
        recent = pd.DataFrame()
    live = live_summary(recent)
    if live.get("samples", 0) == 0:
        st.info(
            "Live accuracy is warming up: no forecast/actual pairs have settled yet. "
            "The dashboard will populate this automatically when Fingrid publishes actual imbalance outcomes. "
            "Until then, the official validation metrics above are the evidence of model accuracy."
        )
    else:
        a, b, c, d, e = st.columns(5)
        a.metric("Live MAE ↓", f"{live['mae']:.2f} €/MWh")
        b.metric("Live coverage", f"{100*live['coverage']:.1f}%")
        c.metric("Within ±€20", f"{100*live['within_20']:.1f}%")
        d.metric("Evaluated rows", f"{live['samples']:,}")
        e.metric("Evaluated origins", f"{live['origins']:,}")
        lp = live_step_performance(recent)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=lp["step"], y=lp["mae"], mode="lines+markers", name="Live MAE"))
        fig.update_layout(title=f"Live MAE by 15-minute horizon – last {live_hours}h", xaxis_title="Step", yaxis_title="MAE EUR/MWh")
        st.plotly_chart(fig, width="stretch")
        settled = recent.copy()
        settled["delivery_start_utc"] = pd.to_datetime(settled["delivery_start_utc"], utc=True)
        settled = settled.sort_values("delivery_start_utc").tail(300)
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=settled["delivery_start_utc"], y=settled["imbalance_price"], name="Actual", mode="lines"))
        fig3.add_trace(go.Scatter(x=settled["delivery_start_utc"], y=settled["p50"], name="Predicted P50", mode="lines"))
        fig3.update_layout(title="Settled actual vs predicted imbalance price", yaxis_title="EUR/MWh", xaxis_title="Delivery UTC")
        st.plotly_chart(fig3, width="stretch")

with learning_tab:
    st.header("Daily self-learning and model governance")
    st.caption("Hourly: adapt bias/fusion/calibration. Daily: create an accuracy checkpoint and conditionally train a challenger. Only a better challenger is promoted.")
    try:
        state = AdaptiveState.load()
        rows = []
        for key, w in state.windows.items():
            rows.append({
                "window": {24:"24h fast",168:"7d regime",336:"14d stable"}.get(w.hours, f"{w.hours}h"),
                "hours": w.hours, "samples": w.samples, "MAE": w.rolling_mae,
                "coverage": w.rolling_coverage, "adapted_15min_horizons": len(w.steps),
            })
        if rows:
            st.dataframe(pd.DataFrame(rows).sort_values("hours"), width="stretch", hide_index=True)
        a, b, c = st.columns(3)
        a.metric("Blended adapted horizons", len(state.blended_steps))
        b.metric("Drift score", f"{state.drift_score:.2f}")
        c.metric("Drift detected", "YES" if state.drift_detected else "No")
        if state.drift_detected:
            st.warning("; ".join(state.drift_reasons))
    except Exception as exc:
        st.warning(f"Adaptive state unavailable: {exc}")

    report = latest_daily_report()
    st.subheader("Latest daily learning checkpoint")
    if not report:
        st.info("No daily checkpoint has been written yet. Run `python scripts/run_daily_learning.py --force --no-retrain` to create one now, or leave the daemon running until the configured daily time.")
    else:
        a, b, c, d = st.columns(4)
        a.metric("Checkpoint date", str(report.get("local_date", "n/a")))
        p24 = report.get("performance_24h", {})
        b.metric("24h settled samples", int(p24.get("samples", 0)))
        c.metric("24h MAE", "n/a" if p24.get("mae") is None else f"{p24['mae']:.2f} €/MWh")
        d.metric("Model action", "promoted" if report.get("retrain", {}).get("promoted") else "kept champion")
        st.write("**Retraining decision:**", report.get("retrain", {}).get("reason", "n/a"))
        with st.expander("Daily learning report JSON"):
            st.json(report)

    hist = daily_history()
    if not hist.empty:
        st.subheader("Daily accuracy trend")
        fig = go.Figure()
        if "mae_24h" in hist:
            fig.add_trace(go.Scatter(x=hist["local_date"], y=hist["mae_24h"], name="24h MAE", mode="lines+markers"))
        if "mae_7d" in hist:
            fig.add_trace(go.Scatter(x=hist["local_date"], y=hist["mae_7d"], name="7d MAE", mode="lines+markers"))
        if "mae_14d" in hist:
            fig.add_trace(go.Scatter(x=hist["local_date"], y=hist["mae_14d"], name="14d MAE", mode="lines+markers"))
        fig.update_layout(yaxis_title="MAE EUR/MWh", xaxis_title="Helsinki date")
        st.plotly_chart(fig, width="stretch")

with signals_tab:
    st.header("Signals and explainability")
    st.caption("Global importance shows which feature families the trained models rely on. Live signal charts show the exact source data available to the current production cycle. Importance is associative model evidence, not proof of causality.")

    try:
        imp = grouped_model_importance(default_model_path())
    except Exception as exc:
        imp = pd.DataFrame()
        st.warning(f"Model importance unavailable: {exc}")
    if not imp.empty:
        c1, c2 = st.columns(2)
        with c1:
            figi = go.Figure(go.Bar(x=100*imp["p50_importance"], y=imp["source_group"], orientation="h"))
            figi.update_layout(title="P50 price-model contribution groups", xaxis_title="Normalized global importance %", yaxis_title="", height=520)
            st.plotly_chart(figi, width="stretch")
        with c2:
            figd = go.Figure(go.Bar(x=100*imp["direction_importance"], y=imp["source_group"], orientation="h"))
            figd.update_layout(title="UP / DOWN / NONE direction-model contribution groups", xaxis_title="Normalized global importance %", yaxis_title="", height=520)
            st.plotly_chart(figd, width="stretch")
        with st.expander("Importance table"):
            show=imp.copy(); show["p50_importance_%"]=100*show.pop("p50_importance"); show["direction_importance_%"]=100*show.pop("direction_importance")
            st.dataframe(show, width="stretch", hide_index=True)

    st.subheader("Explain one live 15-minute prediction")
    latest_origin=str(forecast_all["forecast_origin_utc"].drop_duplicates().iloc[-1])
    explain_step=st.slider("15-minute step to explain",1,40,1,key="explain_step")
    if forecast_path == ONLINE_FORECAST:
        try:
            origin_ts=pd.Timestamp(latest_origin)
            if origin_ts.tzinfo is None: origin_ts=origin_ts.tz_localize("UTC")
            else: origin_ts=origin_ts.tz_convert("UTC")
            ef=build_features(pd.DatetimeIndex([origin_ts]),include_labels=False,include_online_context=True).reset_index(drop=True)
            erow=ef[ef["step"].astype(int)==int(explain_step)].reset_index(drop=True)
            local=local_lgb_group_contributions(default_model_path(),erow,0)
            if not local.empty:
                local=local.head(12)
                figx=go.Figure()
                figx.add_trace(go.Bar(name="Direct-price LightGBM",x=local["source_group"],y=local["direct_price_eur_mwh"]))
                figx.add_trace(go.Bar(name="Spread LightGBM",x=local["source_group"],y=local["spread_eur_mwh"]))
                figx.update_layout(barmode="group",title=f"Signed model evidence for step {explain_step} (+{(explain_step-1)*15} min)",yaxis_title="Approx. contribution (EUR/MWh)",xaxis_title="Feature family",height=430)
                st.plotly_chart(figx,width="stretch")
                st.caption("Positive bars push that LightGBM expert upward; negative bars push it downward. This is model evidence, not proof of causality. The final P50 also includes CatBoost, direction gating, calibration and online adaptation.")
            else:
                st.info("Local tree contributions are not available for the active model.")
        except Exception as exc:
            st.warning(f"Local explanation unavailable: {exc}")
    else:
        st.info("Local live explanation is shown after running one V10 online cycle.")

    st.subheader("Separate live graphs by contributing source")
    if LIVE_LONG_PARQUET.exists():
        try:
            raw=pd.read_parquet(LIVE_LONG_PARQUET)
            raw["timestamp_utc"]=pd.to_datetime(raw["timestamp_utc"],utc=True,errors="coerce")
            raw=raw.dropna(subset=["timestamp_utc","value"])
        except Exception as exc:
            raw=pd.DataFrame(); st.warning(f"Live source cache could not be read: {exc}")
    else:
        raw=pd.DataFrame()
    if raw.empty:
        st.info("Run one online cycle to populate live source graphs.")
    else:
        for group in LIVE_SIGNAL_GROUPS:
            g=select_live_group(raw,group)
            if g.empty:
                continue
            st.markdown(f"#### {group}")
            # Plot one chart per unit so MW, EUR/MWh, Hz, and state codes are never mixed on one axis.
            for unit, u in g.groupby(g["unit"].fillna("")):
                figg=go.Figure()
                for sid, ss in u.groupby("series_id"):
                    ss=ss.sort_values("timestamp_utc")
                    figg.add_trace(go.Scatter(x=ss["timestamp_utc"], y=ss["value"], mode="lines", name=str(sid)))
                figg.update_layout(height=300, xaxis_title="UTC", yaxis_title=str(unit) if str(unit) else "value", legend_title="series")
                st.plotly_chart(figg, width="stretch")

    st.subheader("How to read one prediction")
    if not forecast_all.empty:
        current=forecast_all[forecast_all["forecast_origin_utc"]==forecast_all["forecast_origin_utc"].iloc[-1]].copy()
        if not current.empty:
            row=current.iloc[0]
            parts=[
                ("Main forecast", f"P50 {row.get('p50', float('nan')):.2f} €/MWh"),
                ("Uncertainty", f"P10 {row.get('p10', float('nan')):.2f} to P90 {row.get('p90', float('nan')):.2f} €/MWh"),
                ("Direction", f"UP {100*row.get('p_up',0):.0f}% / DOWN {100*row.get('p_down',0):.0f}% / NONE {100*row.get('p_none',0):.0f}%"),
                ("Model disagreement", "n/a" if pd.isna(row.get('model_disagreement',np.nan)) else f"{row.get('model_disagreement'):.2f} €/MWh"),
                ("Reliability", "n/a" if pd.isna(row.get('reliability',np.nan)) else f"{100*row.get('reliability'):.0f}%"),
                ("Top transparent drivers", str(row.get('top_drivers','n/a'))),
            ]
            st.dataframe(pd.DataFrame(parts,columns=["explanation","reading"]), width="stretch", hide_index=True)

with replay_tab:
    st.header("Historical validation replay")
    detail = validation_detail()
    if detail.empty:
        st.warning("Historical replay needs `data/raw/starter_package_v1/data/actuals_validation.csv` plus the V3 validation forecast.")
    else:
        replay_origins = detail["forecast_origin_utc"].drop_duplicates().sort_values().tolist()
        selected = st.select_slider("Move the point of review into the past", replay_origins, value=replay_origins[-1])
        r = detail[detail["forecast_origin_utc"] == selected].copy()
        r["delivery_local"] = r["delivery_start_utc"].dt.tz_convert(HELSINKI)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=r.delivery_local, y=r.p90, line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=r.delivery_local, y=r.p10, fill="tonexty", name="P10–P90", line=dict(width=0)))
        fig.add_trace(go.Scatter(x=r.delivery_local, y=r.p50, name="Forecast P50", mode="lines+markers"))
        fig.add_trace(go.Scatter(x=r.delivery_local, y=r.imbalance_price, name="Actual imbalance price", mode="lines+markers"))
        fig.update_layout(title="What the system forecast then vs what actually happened", yaxis_title="EUR/MWh", xaxis_title="Delivery Europe/Helsinki")
        st.plotly_chart(fig, width="stretch")
        valid = r[r["abs_error"].notna()]
        if not valid.empty:
            a, b, c = st.columns(3)
            a.metric("This origin MAE", f"{valid.abs_error.mean():.2f} €/MWh")
            b.metric("Coverage", f"{100*valid.covered.mean():.1f}%")
            c.metric("Worst absolute error", f"{valid.abs_error.max():.2f} €/MWh")
        cols = ["step", "delivery_local", "p10", "p50", "p90", "imbalance_price", "abs_error", "covered", "p_up", "p_down", "p_none", "p_spike_high", "p_spike_low", "top_drivers"]
        st.dataframe(r[[c for c in cols if c in r]], width="stretch", hide_index=True, height=650)

with health_tab:
    st.header("Data health and Finnish power-market digital twin")
    try:
        active_path=default_model_path(); active_bundle=load_any_bundle(active_path)
        st.subheader("Active production model")
        a,b=st.columns(2)
        a.metric("Model version", str(getattr(active_bundle,"version",active_path.stem)))
        b.metric("Model source", "online champion" if "models/online/champion" in str(active_path).replace("\\","/") else "packaged default")
        st.caption(f"Loaded from: {active_path}")
    except Exception as exc:
        st.warning(f"Could not inspect active model: {exc}")
    if ONLINE_MARKET_STATE_JSON.exists():
        twin = json.loads(ONLINE_MARKET_STATE_JSON.read_text(encoding="utf-8"))
        a, b, c, d, e = st.columns(5)
        a.metric("Market pressure", str(twin.get("market_pressure", "unknown")))
        b.metric("Reserve stress", _fmt(twin.get("reserve_stress_index"), 2))
        c.metric("Cross-border stress", _fmt(twin.get("cross_border_stress"), 2))
        d.metric("24h adaptive MAE", _fmt(twin.get("adaptive_mae_24h"), 1, " €/MWh"))
        e.metric("Drift", "YES" if twin.get("drift_detected") else "No")
        with st.expander("Digital-twin state"):
            st.json(twin)
    else:
        st.info("No market-state snapshot yet.")

    if LIVE_SOURCE_HEALTH_JSON.exists():
        h = json.loads(LIVE_SOURCE_HEALTH_JSON.read_text(encoding="utf-8"))
        st.subheader("Live API source health")
        st.progress(float(h.get("healthy_fraction", 0.0)), text=f"Healthy items: {h.get('items_ok',0)} | errors: {h.get('items_error',0)} | empty: {h.get('items_empty',0)}")
        details = []
        for item in h.get("details", []):
            details.append({
                "source": item.get("source"), "dataset_id": item.get("dataset_id"), "series_id": item.get("series_id"),
                "status": item.get("status"), "rows": item.get("rows"), "error": item.get("error"),
            })
        if details:
            st.dataframe(pd.DataFrame(details), width="stretch", hide_index=True)
        with st.expander("Raw source-health JSON"):
            st.json(h)
    else:
        st.info("Run a live online cycle to populate source health.")

st.caption(
    "V11 keeps the official hackathon backtest frozen and leakage-safe. The V10 full-refit model remains the champion until a V11 challenger wins on an untouched live evaluation tail. ENTSO-E MCP/revision features are live/post-cutoff upgrades; production learning uses only settled outcomes and exact prediction-time feature snapshots."
)
