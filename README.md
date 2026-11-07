# TE-BalanceAI V11

See `README_V11.md` for the current production upgrade, active-model checks, challenger tournament, and run commands.

# TE-BalanceAI V10

Current system: **V10 production P50 refit + robust direct/spread fusion + forecast revisions + sample-aware reliability + asymmetric live conformal calibration + expanded ENTSO-E MCP + explainability**.

Start with `README_V10.md`.

Locked organizer-style validation evidence remains **43.4074 EUR/MWh P50 MAE, 12.4193 mean P10/P90 pinball, 79.75% coverage, 0.0973 spike Brier, 0.4214 direction Brier**. V10's final production model is trained on those historical validation labels, so that locked score remains the honest competition evidence rather than rescoring training data.

A strict chronological production-refit experiment improved final-block MAE from **34.16 to 30.82 EUR/MWh (9.77% lower)**. Hidden-test and future live MAE remain unknown until actual outcomes settle.
