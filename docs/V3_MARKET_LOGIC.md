# V3 Market Logic - Turku Energia 2025 Statement -> Forecast Features

This note documents how public business/market information from Turku Energia's 2025 financial statement is used without feeding annual accounting figures into a 15-minute forecasting model.

## Used as model/design logic

1. **Weather-dependent renewable regime** - the statement describes increasing wind generation and more weather-dependent electricity price behaviour. V3 therefore strengthens wind-ramp, renewable-share, net-load and volatility features.
2. **Cross-border transmission stress** - the statement discusses Estlink 2 and Aurora Line effects on Finnish area prices. V3 therefore adds net export, flow reversal, FI-EE directional capacity utilisation and composite cross-border stress.
3. **Nordic/hydro context** - the statement describes Nordic hydrological conditions as relevant to price levels. The starter package does not contain reservoir history, so V3 uses neighbouring imbalance prices and NO4 flow only as an explicit `nordic_market_pressure_proxy`. It does not pretend this is measured reservoir storage.
4. **Flexible electric load / electric boilers** - the statement describes rapidly growing Finnish electric-boiler capacity and the possibility that flexible electric demand reduces the depth of low-price periods. V3 adds a low-spot/flexible-load regime signal. It does not claim to know Turku Energia's actual boiler dispatch.
5. **Volume and forecast risk** - the statement identifies purchase/sales price variation and volume/profile risk. V3 strengthens recent consumption forecast error, balance, reserve activation and direction-persistence features.
6. **Regional weather dashboard** - Pori, Kalajoki and Raahe are included among live weather nodes because the statement reports Turku Energia wind interests in those regions. The dashboard weights are equal context weights, not capacity shares.

## Deliberately NOT used as 15-minute model inputs

- annual revenue
- profit/loss
- debt
- cash flow
- balance-sheet totals
- annual derivative nominal values

These are important corporate context but are not timestamped short-horizon causal inputs for the challenge target.

## External data integrity rule

Live weather is dashboard/context-only until historical forecast vintages can be reconstructed with a defensible publication timestamp. The scored backtest continues to enforce the organizer's row-level `first_usable_origin_utc` rule.
