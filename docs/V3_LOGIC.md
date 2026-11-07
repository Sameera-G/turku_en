# TE-BalanceAI V3 logic

V3 translates useful operational findings from Turku Energia's 2025 financial statement into short-horizon market features. It does **not** use annual revenue, profit, debt or balance-sheet values as forecasting inputs.

## Added market logic

1. **Weather / renewable dependence**
   - wind and solar forecast ramps
   - renewable-share proxy
   - net-load proxy
   - wind-ramp stress

2. **Cross-border sensitivity**
   - net export and change in net export
   - gross border flow
   - FI-EE directional capacity-utilisation proxy
   - capacity-stress flag
   - border-flow reversal

3. **Nordic market context**
   - neighbouring imbalance-price mean and dispersion
   - Finland-vs-neighbour price spread
   - NO4-flow-assisted Nordic market-pressure proxy
   - This is explicitly a market proxy, not a true hydro-reservoir measurement.

4. **Flexible electric load / sector coupling**
   - low-spot signal
   - negative spot flag
   - sector-coupling / electric-boiler opportunity proxy
   - These are market-state indicators, not plant-dispatch instructions.

5. **Volume / forecast-error risk**
   - recent production-consumption balance
   - recent load forecast error and percentage error
   - reserve activation and direction persistence

6. **Price reliability**
   - LightGBM and CatBoost median forecasts
   - horizon-specific late-fusion weight learned on validation MAE
   - model disagreement becomes part of the reliability score
   - P10/P90 remain horizon-calibrated

## Leakage rule

Every feature lookup still enforces:

`first_usable_origin_utc <= forecast_origin_utc`

Test-period observations can become inference context only after they are published. Model parameters remain frozen during hidden-test replay.

## Important boundary

Live regional weather downloaded by `scripts/download_public_context.py --live` is for the live dashboard / production demo only. It is not used in the hidden historical backtest unless a historical forecast-vintage source with auditable publication time is added.
