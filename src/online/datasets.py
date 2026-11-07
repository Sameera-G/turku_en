from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: int
    series_id: str
    unit: str
    kind: str  # actual | forecast | context
    target_minutes: int = 15


# Dataset IDs are intentionally explicit and easy to audit. Core challenge IDs
# are preserved where possible; live 15-minute forecasts use Fingrid's updated
# forecast products (166 / 245 / 248) but are mapped to the canonical feature
# names expected by the V3/V4 forecasting model.
FINGRID_DATASETS: tuple[DatasetSpec, ...] = (
    DatasetSpec(319, "fi_imbalance_price", "EUR/MWh", "actual"),
    DatasetSpec(369, "fi_mfrr_direction", "-1/0/1", "actual"),
    DatasetSpec(375, "fi_mfrr_up_activated", "MW", "actual"),
    DatasetSpec(376, "fi_mfrr_down_activated", "MW", "actual"),
    DatasetSpec(244, "fi_mfrr_up_price", "EUR/MWh", "actual"),
    DatasetSpec(106, "fi_mfrr_down_price", "EUR/MWh", "actual"),
    DatasetSpec(347, "fi_afrr_up_price", "EUR/MWh", "actual"),
    DatasetSpec(348, "fi_afrr_down_price", "EUR/MWh", "actual"),
    DatasetSpec(124, "fi_consumption", "MW", "actual"),
    DatasetSpec(74, "fi_production", "MW", "actual"),
    DatasetSpec(55, "fi_ee_flow", "MW", "actual"),
    DatasetSpec(57, "fi_no4_flow", "MW", "actual"),
    DatasetSpec(60, "fi_se1_flow", "MW", "actual"),
    DatasetSpec(61, "fi_se3_flow", "MW", "actual"),
    # Updated live forecasts. These are deliberately overlaid on the canonical
    # day-ahead feature names for the current production origin only. Exact
    # feature snapshots are persisted, so online retraining never reconstructs
    # past origins from later forecast revisions.
    DatasetSpec(166, "fi_consumption_fc_da", "MW", "forecast"),
    DatasetSpec(245, "fi_wind_fc_da", "MWh/h", "forecast"),
    DatasetSpec(248, "fi_solar_fc_da", "MWh/h", "forecast"),
    # Operational context. These are stored and shown in diagnostics; they are
    # not injected into the frozen V3 model unless a future challenger model is
    # explicitly trained with them.
    DatasetSpec(198, "fi_surplus_deficit_rt", "MW", "context"),
    DatasetSpec(371, "fi_electric_boiler_rt", "MW", "context"),
    DatasetSpec(177, "fi_frequency_rt", "Hz", "context"),
    DatasetSpec(75, "fi_wind_generation_rt", "MW", "context"),
    DatasetSpec(191, "fi_hydro_generation_rt", "MW", "context"),
    DatasetSpec(188, "fi_nuclear_generation_rt", "MW", "context"),
    # V8 additional public operational context.  These do not enter the locked
    # hackathon backtest; exact live snapshots are stored and may be admitted to
    # a future challenger only after sufficient coverage exists.
    DatasetSpec(241, "fi_production_fc_rt", "MW", "forecast"),
    DatasetSpec(268, "fi_wind_capacity_rt", "MW", "context"),
    DatasetSpec(186, "fi_surplus_deficit_cumulative_rt", "MWh", "context"),
    DatasetSpec(201, "fi_chp_district_heat_rt", "MW", "context"),
    DatasetSpec(202, "fi_chp_industrial_rt", "MW", "context"),
    DatasetSpec(183, "fi_peak_load_power_rt", "MW", "context"),
    DatasetSpec(398, "fi_bess_discharge_rt", "MW", "context"),
    DatasetSpec(336, "fi_shortage_status_rt", "0-3", "context"),
    DatasetSpec(30, "fi_north_south_flow_rt", "MW", "context"),
    DatasetSpec(209, "fi_power_system_state_rt", "1-5", "context"),
    DatasetSpec(194, "fi_net_import_export_rt", "MW", "context"),
)


BY_ID = {x.dataset_id: x for x in FINGRID_DATASETS}
BY_SERIES = {x.series_id: x for x in FINGRID_DATASETS}
