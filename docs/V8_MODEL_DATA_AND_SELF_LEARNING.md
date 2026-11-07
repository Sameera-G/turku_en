# V8 engineering note

## 1. Why spot-spread modeling

Turku Energia's challenge material explains that the dominant balancing direction is central to the imbalance-price formation: upward regulation is generally above spot, downward regulation below spot, and no dominant mFRR direction is spot-based. V8 therefore predicts the deviation from spot as a first-class target instead of asking one black-box model to learn the entire price level.

## 2. P50 architecture

For delivery period h:

- `S_h`: legally known Finnish day-ahead price;
- `D_h`: DOWN spread expert;
- `U_h`: UP spread expert;
- NONE spread = 0;
- `p_down, p_none, p_up`: calibrated direction probabilities.

A sharpened probability gate produces:

`gated_price = S_h + p_down*D_h + p_up*U_h`

That is blended with global LightGBM/CatBoost spread models and, when validation supports it, a small amount of the direct-price model. Each horizon has its own fusion settings. P10/P90 and the challenge probability heads remain separately calibrated.

## 3. Public-validation evidence

Exact organizer scorer result:

- MAE 43.4074 EUR/MWh
- mean P10/P90 pinball 12.4193
- P10-P90 coverage 79.75%
- exceptional-price Brier 0.0973
- direction Brier 0.4214

V3 MAE was 47.2047 EUR/MWh. Organizer public-validation baseline MAE was 66.20 EUR/MWh.

## 4. Direction is the next major opportunity

A diagnostic using true direction only as an oracle (never used in submitted forecasts) showed much lower possible MAE, demonstrating that better UP/DOWN/NONE probability forecasts could substantially improve the spread gate. V8 keeps the validated direction head; a later experiment can ensemble an additional direction learner if it improves chronological holdout Brier and hidden-test robustness.

## 5. Added operational data

V8 live mode adds high-value Fingrid signals including updated total production forecast, wind forecast capacity, BESS discharge, CHP production, shortage state and north-south transfer. These are production-only until enough exact prediction-time snapshots have accumulated for challenger training.

## 6. Day-ahead spot source hierarchy

- preferred: ENTSO-E Transparency Platform token;
- preferred where licensed: Nord Pool feed/manual licensed curve;
- hackathon/demo fallback: no-key Finnish public price API, clearly labelled and attributed;
- if absent: direct-price V3 fallback, with reduced reliability.

No website scraping is used.

## 7. Weather research path

Open-Meteo Single Runs is appropriate for leakage-safe weather backtesting because every archived run is queryable by initialization time. A conservative publication lag must still be applied because initialization time is not publication time. FMI operational forecasts/observations remain useful for live cross-checks and forecast-error features.

## 8. Self-learning

Hourly:

1. ingest newest legally available data;
2. settle predictions only when actuals are available after the configured lag;
3. calculate errors;
4. update 24h / 7d / 14d horizon-specific bias, direct-vs-spread weight, interval correction and probability calibration;
5. detect drift;
6. issue the next 40 quarter-hour forecasts.

Daily:

1. record 24h/7d/14d accuracy;
2. if enough settled data and spot labels exist, train a chronological challenger;
3. train both direct probabilistic base and V8 spread experts;
4. compare MAE, pinball, direction Brier and spike Brier;
5. reject if gain is insufficient or any protected metric regresses too much;
6. promote only the better challenger and retain versioned rollback artifacts.

## 9. Self-learning verification

The isolated smoke test uses only public validation data and a temporary database. It has been run successfully with non-zero settled evaluations and adapted horizons, including learned direct-vs-spread weights. This verifies the plumbing; it does not claim that a few hours of live data will automatically improve real-world accuracy.
