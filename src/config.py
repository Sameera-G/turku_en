from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_EXTERNAL = DATA_RAW / "external"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
REPORTS_DIR = ROOT / "reports"

STARTER_DIR = DATA_RAW / "starter_package_v1"
INPUTS_CSV = STARTER_DIR / "data" / "inputs.csv"
INPUTS_PARQUET = STARTER_DIR / "data" / "inputs.parquet"
SPOT_CSV = STARTER_DIR / "data" / "spot_nordpool.csv"
ACTUALS_VALIDATION_CSV = STARTER_DIR / "data" / "actuals_validation.csv"
FORECAST_ORIGINS_CSV = STARTER_DIR / "forecast_origins.csv"
SAMPLE_SUBMISSION_CSV = STARTER_DIR / "sample_submission.csv"
VALIDATOR = STARTER_DIR / "validate_submission.py"
SCORER = STARTER_DIR / "score.py"
MAKE_TEMPLATES = STARTER_DIR / "make_templates.py"
AVAILABILITY_CSV = STARTER_DIR / "availability.csv"

# Organizer-defined chronology, UTC.
TEST_START_UTC = "2026-10-04T21:00:00Z"       # 5 Oct 00:00 Europe/Helsinki
VALIDATION_START_UTC = "2026-09-06T21:00:00Z" # 7 Sep 00:00 Europe/Helsinki
TRAIN_LABEL_START_UTC = "2026-06-01T00:00:00Z"

FORECAST_STEPS = 40
STEP_MINUTES = 15
SPIKE_MARGIN = 50.0
RANDOM_SEED = 42
MODEL_THREADS = max(1, int(os.getenv("TE_BALANCEAI_THREADS", "1")))

# Delivery-time values that may be known in advance when the package's
# first_usable_origin_utc says they were already published.
FUTURE_SERIES = [
    "fi_spot_price",
    "fi_consumption_fc_da",
    "fi_wind_fc_da",
    "fi_solar_fc_da",
    "se1_load_fc_da",
    "se3_load_fc_da",
    "ee_load_fc_da",
    "fi_ee_capacity_da",
    "ee_fi_capacity_da",
    "cal_weekday_local",
    "cal_hour_local",
    "cal_is_holiday_fi",
    "cal_is_dst",
]

# Runtime observations. During hidden-test replay these may include earlier
# test-period rows once first_usable_origin_utc <= forecast origin. Model
# parameters remain frozen.
HISTORY_SERIES = [
    "fi_imbalance_price",
    "fi_mfrr_up_price",
    "fi_mfrr_down_price",
    "fi_mfrr_up_activated",
    "fi_mfrr_down_activated",
    "fi_mfrr_direction",
    "fi_afrr_up_price",
    "fi_afrr_down_price",
    "fi_consumption",
    "fi_production",
    "fi_se1_flow",
    "fi_se3_flow",
    "fi_no4_flow",
    "fi_ee_flow",
    "se1_imbalance_price_long",
    "se1_imbalance_price_short",
    "se3_imbalance_price_long",
    "se3_imbalance_price_short",
    "ee_imbalance_price_long",
    "ee_imbalance_price_short",
]

# V3 fixes V2's missing 15-minute and 8-hour snapshots and adds 12h context.
# 1q=15m, 2q=30m, 4q=1h, 8q=2h, 16q=4h, 32q=8h, 48q=12h, 96q=24h.
HISTORY_LAGS_Q = [1, 2, 4, 8, 16, 32, 96]

# Financial-statement-informed production context. These are NOT accounting
# values fed to the model. They identify regions where Turku Energia reports
# wind interests and are used only by the optional live weather downloader/UI.
WEATHER_NODES = {
    # Equal dashboard-context weights: these are NOT installed-capacity shares.
    # Pori/Kalajoki/Raahe are included because the 2025 statement reports wind
    # interests there; Vaasa/Oulu/Turku broaden the weather view.
    "turku": (60.4518, 22.2666, 1/6),
    "pori": (61.4851, 21.7974, 1/6),
    "kalajoki": (64.2600, 23.9500, 1/6),
    "raahe": (64.6833, 24.4833, 1/6),
    "vaasa": (63.0951, 21.6165, 1/6),
    "oulu": (65.0121, 25.4651, 1/6),
}

# ---------------------------------------------------------------------------
# V4 production / continuous-learning configuration.
# This is intentionally separate from the locked hackathon backtest logic.
# ---------------------------------------------------------------------------
ONLINE_DIR = ROOT / "data" / "online"
LIVE_RAW_DIR = DATA_RAW / "live"
LIVE_LONG_PARQUET = LIVE_RAW_DIR / "fingrid_live_long.parquet"
ONLINE_DB = ONLINE_DIR / "te_balanceai_online.sqlite3"
ONLINE_ADAPTIVE_STATE = ONLINE_DIR / "adaptive_state.json"
ONLINE_CHAMPION = MODELS_DIR / "online" / "champion.joblib"
ONLINE_VERSIONS_DIR = MODELS_DIR / "online" / "versions"
ONLINE_OUTPUTS_DIR = OUTPUTS_DIR / "online"
ONLINE_REPORTS_DIR = REPORTS_DIR / "online"
MANUAL_SPOT_CSV = ONLINE_DIR / "manual_spot.csv"
PUBLIC_SPOT_API_BASE = os.getenv("PUBLIC_SPOT_API_BASE", "https://sahkonhintahuomenna.fi/api/v1/fi").strip()
ONLINE_USE_PUBLIC_SPOT_FALLBACK = os.getenv("TE_BALANCEAI_USE_PUBLIC_SPOT_FALLBACK", "1").strip() not in {"0","false","False"}

FINGRID_API_BASE = os.getenv("FINGRID_API_BASE", "https://data.fingrid.fi/api").rstrip("/")
FINGRID_API_KEY = os.getenv("FINGRID_API_KEY", "").strip()
ENTSOE_API_TOKEN = os.getenv("ENTSOE_API_TOKEN", "").strip()
ENTSOE_API_BASE = os.getenv("ENTSOE_API_BASE", "https://web-api.tp.entsoe.eu/api").strip()
ENTSOE_MCP_API_BASE = os.getenv("ENTSOE_MCP_API_BASE", "https://entsoemcp.com/api/series").strip()
ENTSOE_MCP_URL = os.getenv("ENTSOE_MCP_URL", "https://entsoemcp.com/mcp").strip()
ONLINE_USE_ENTSOE_MCP = os.getenv("TE_BALANCEAI_USE_ENTSOE_MCP", "1").strip() not in {"0","false","False"}
ENTSOE_MCP_TIMEOUT_SECONDS = float(os.getenv("TE_BALANCEAI_ENTSOE_MCP_TIMEOUT_SECONDS", "8"))

# The production daemon runs shortly after each full hour and forecasts that
# nominal current hourly origin. The delay lets newly published source data land.
ONLINE_CYCLE_SECOND = int(os.getenv("TE_BALANCEAI_CYCLE_SECOND", "75"))
ONLINE_LOOKBACK_HOURS = int(os.getenv("TE_BALANCEAI_LIVE_LOOKBACK_HOURS", "36"))
ONLINE_FORECAST_HOURS = int(os.getenv("TE_BALANCEAI_LIVE_FORECAST_HOURS", "72"))
ONLINE_EVAL_HOURS = int(os.getenv("TE_BALANCEAI_EVAL_HOURS", "24"))
ONLINE_SETTLEMENT_LAG_MINUTES = int(os.getenv("TE_BALANCEAI_SETTLEMENT_LAG_MINUTES", "75"))
ONLINE_FULL_RETRAIN_HOURS = int(os.getenv("TE_BALANCEAI_FULL_RETRAIN_HOURS", "24"))
ONLINE_MIN_RETRAIN_ORIGINS = int(os.getenv("TE_BALANCEAI_MIN_RETRAIN_ORIGINS", "48"))
ONLINE_RETRAIN_LOOKBACK_DAYS = int(os.getenv("TE_BALANCEAI_RETRAIN_LOOKBACK_DAYS", "90"))
ONLINE_PROMOTION_MIN_GAIN = float(os.getenv("TE_BALANCEAI_PROMOTION_MIN_GAIN", "0.005"))
ONLINE_PROMOTION_MIN_MAE_GAIN = float(os.getenv("TE_BALANCEAI_PROMOTION_MIN_MAE_GAIN", "0.002"))
ONLINE_MAX_METRIC_REGRESSION = float(os.getenv("TE_BALANCEAI_MAX_METRIC_REGRESSION", "0.05"))
# V11 uses separate chronological calibration and untouched evaluation tails
# for challenger promotion. With the default 48-origin minimum this leaves
# at least 24 online origins for training and 12+12 for calibration/evaluation.
ONLINE_RETRAIN_CAL_ORIGINS = int(os.getenv("TE_BALANCEAI_RETRAIN_CAL_ORIGINS", "12"))
ONLINE_RETRAIN_EVAL_ORIGINS = int(os.getenv("TE_BALANCEAI_RETRAIN_EVAL_ORIGINS", "12"))
ONLINE_ENABLE_STRONG_P50_CHALLENGER = os.getenv("TE_BALANCEAI_ENABLE_STRONG_P50_CHALLENGER", "1").strip() not in {"0","false","False"}
ONLINE_ENABLE_FULL_RETRAIN = os.getenv("TE_BALANCEAI_ENABLE_FULL_RETRAIN", "1").strip() not in {"0", "false", "False"}

# V5 live multi-source settings. These affect only production/continuous mode.
OPENMETEO_API_BASE = os.getenv("OPENMETEO_API_BASE", "https://api.open-meteo.com/v1/forecast").strip()
FMI_WFS_BASE = os.getenv("FMI_WFS_BASE", "https://opendata.fmi.fi/wfs").strip()
ONLINE_USE_OPENMETEO = os.getenv("TE_BALANCEAI_USE_OPENMETEO", "1").strip() not in {"0","false","False"}
ONLINE_USE_FMI = os.getenv("TE_BALANCEAI_USE_FMI", "1").strip() not in {"0","false","False"}
ONLINE_USE_ENTSOE_EXTENDED = os.getenv("TE_BALANCEAI_USE_ENTSOE_EXTENDED", "1").strip() not in {"0","false","False"}
ONLINE_SOURCE_STALE_MINUTES = int(os.getenv("TE_BALANCEAI_SOURCE_STALE_MINUTES", "90"))
ONLINE_DAYAHEAD_REFRESH_LOCAL_HOUR = int(os.getenv("TE_BALANCEAI_DAYAHEAD_REFRESH_LOCAL_HOUR", "14"))
ONLINE_RETRAIN_EXTRA_MIN_COVERAGE = float(os.getenv("TE_BALANCEAI_RETRAIN_EXTRA_MIN_COVERAGE", "0.35"))
ONLINE_ARCHIVE_RAW = os.getenv("TE_BALANCEAI_ARCHIVE_RAW", "1").strip() not in {"0","false","False"}
LIVE_SOURCE_HEALTH_JSON = ONLINE_DIR / "source_health.json"
LIVE_CONTEXT_JSON = ONLINE_DIR / "latest_context.json"
LIVE_RAW_SNAPSHOT_DIR = ONLINE_DIR / "raw_snapshots"

# Extra live-only series. The locked V3 hackathon backtest does not request
# these columns. V5 passes include_online_context=True only in production mode.
ONLINE_FUTURE_SERIES = [
    "fi_production_fc_rt",
    "weather_wind_100m_weighted", "weather_wind_10m_weighted", "weather_wind_gust_max", "weather_temperature_mean",
    "mcp_fi_day_ahead_price", "mcp_fi_intraday_price", "mcp_fi_load_forecast", "mcp_fi_generation_forecast",
    "mcp_fi_wind_forecast", "mcp_fi_solar_forecast",
    "mcp_se1_load_forecast", "mcp_se3_load_forecast", "mcp_ee_load_forecast", "mcp_no4_load_forecast",
    "mcp_se1_day_ahead_price", "mcp_se3_day_ahead_price", "mcp_ee_day_ahead_price", "mcp_no4_day_ahead_price",
    "mcp_fi_afrr_up_capacity_price", "mcp_fi_afrr_down_capacity_price", "mcp_fi_afrr_up_capacity_mw", "mcp_fi_afrr_down_capacity_mw",
    "mcp_fi_afrr_up_procured_mw", "mcp_fi_afrr_down_procured_mw",
    "mcp_fi_generation_unavailable_mw",
    "mcp_sched_fi_se1", "mcp_sched_se1_fi", "mcp_sched_fi_se3", "mcp_sched_se3_fi",
    "mcp_sched_fi_ee", "mcp_sched_ee_fi", "mcp_sched_fi_no4", "mcp_sched_no4_fi",
    "mcp_ntc_fi_se1", "mcp_ntc_se1_fi", "mcp_ntc_fi_se3", "mcp_ntc_se3_fi",
    "mcp_ntc_fi_ee", "mcp_ntc_ee_fi", "mcp_ntc_fi_no4", "mcp_ntc_no4_fi",
    "weather_pressure_mean", "weather_precipitation_mean", "weather_cloud_mean",
    "weather_solar_mean", "entsoe_load_forecast", "entsoe_wind_forecast",
    "entsoe_solar_forecast", "entsoe_hydro_reservoir", "entsoe_generation_unavailable_mw",
    "entsoe_transmission_outage_count",
    "revision__fi_spot_price", "revision__fi_consumption_fc_da", "revision__fi_wind_fc_da",
    "revision__fi_solar_fc_da", "revision__fi_production_fc_rt",
    "revision__mcp_fi_day_ahead_price", "revision__mcp_fi_intraday_price",
    "revision__mcp_fi_load_forecast", "revision__mcp_fi_generation_forecast",
    "revision__mcp_fi_wind_forecast", "revision__mcp_fi_solar_forecast",
    "revision__weather_wind_100m_weighted", "revision__weather_temperature_mean",
]
ONLINE_HISTORY_SERIES = [
    "fi_surplus_deficit_cumulative_rt", "fi_chp_district_heat_rt", "fi_chp_industrial_rt",
    "fi_peak_load_power_rt", "fi_bess_discharge_rt", "fi_shortage_status_rt", "fi_north_south_flow_rt", "fi_wind_capacity_rt",
    "fi_surplus_deficit_rt", "fi_electric_boiler_rt", "fi_frequency_rt",
    "fi_wind_generation_rt", "fi_hydro_generation_rt", "fi_nuclear_generation_rt",
    "entsoe_actual_load", "entsoe_wind_generation", "entsoe_solar_generation",
    "mcp_fi_imbalance_volume",
    "mcp_fi_afrr_up_activated_price", "mcp_fi_afrr_down_activated_price",
    "mcp_fi_mfrr_up_activated_price", "mcp_fi_mfrr_down_activated_price",
    "mcp_ee_imbalance_long", "mcp_ee_imbalance_short", "mcp_se1_imbalance_long", "mcp_se1_imbalance_short",
    "mcp_se3_imbalance_long", "mcp_se3_imbalance_short", "mcp_no4_imbalance_long", "mcp_no4_imbalance_short",
    "fi_power_system_state_rt", "fi_net_import_export_rt",
    "entsoe_hydro_generation", "entsoe_nuclear_generation",
    "weather_obs_temperature_mean", "weather_obs_wind_10m_weighted",
    "weather_obs_pressure_mean", "weather_obs_precipitation_mean", "weather_obs_cloud_mean",
]


# ---------------------------------------------------------------------------
# V6 Helen-inspired adaptive learning / digital-twin configuration.
# These settings affect production mode only; the locked hackathon backtest
# remains unchanged.
# ---------------------------------------------------------------------------
ONLINE_ADAPT_WINDOWS_HOURS = tuple(
    int(x.strip()) for x in os.getenv("TE_BALANCEAI_ADAPT_WINDOWS_HOURS", "24,168,336").split(",") if x.strip()
)
_w = [float(x.strip()) for x in os.getenv("TE_BALANCEAI_ADAPT_WINDOW_WEIGHTS", "0.50,0.30,0.20").split(",") if x.strip()]
if len(_w) != len(ONLINE_ADAPT_WINDOWS_HOURS):
    _w = [1.0 / len(ONLINE_ADAPT_WINDOWS_HOURS)] * len(ONLINE_ADAPT_WINDOWS_HOURS)
_s = sum(_w) or 1.0
ONLINE_ADAPT_WINDOW_WEIGHTS = {h: w / _s for h, w in zip(ONLINE_ADAPT_WINDOWS_HOURS, _w)}
ONLINE_DRIFT_MAE_RATIO = float(os.getenv("TE_BALANCEAI_DRIFT_MAE_RATIO", "1.35"))
ONLINE_DRIFT_COVERAGE_DEVIATION = float(os.getenv("TE_BALANCEAI_DRIFT_COVERAGE_DEVIATION", "0.12"))
ONLINE_DRIFT_MIN_SAMPLES = int(os.getenv("TE_BALANCEAI_DRIFT_MIN_SAMPLES", "24"))
ONLINE_DRIFT_RETRAIN_COOLDOWN_HOURS = int(os.getenv("TE_BALANCEAI_DRIFT_RETRAIN_COOLDOWN_HOURS", "6"))
ONLINE_MARKET_STATE_JSON = ONLINE_DIR / "latest_market_state.json"
ONLINE_WEATHER_ERROR_MIN_SAMPLES = int(os.getenv("TE_BALANCEAI_WEATHER_ERROR_MIN_SAMPLES", "8"))
ONLINE_CONFORMAL_TARGET_COVERAGE = float(os.getenv("TE_BALANCEAI_CONFORMAL_TARGET_COVERAGE", "0.80"))
ONLINE_CONFORMAL_MIN_SAMPLES = int(os.getenv("TE_BALANCEAI_CONFORMAL_MIN_SAMPLES", "8"))
ONLINE_CONFORMAL_FULL_STRENGTH_SAMPLES = int(os.getenv("TE_BALANCEAI_CONFORMAL_FULL_STRENGTH_SAMPLES", "48"))
ONLINE_CONFORMAL_MAX_EXPAND_EUR = float(os.getenv("TE_BALANCEAI_CONFORMAL_MAX_EXPAND_EUR", "250"))
ONLINE_GLOBAL_ADAPT_MIN_SAMPLES = int(os.getenv("TE_BALANCEAI_GLOBAL_ADAPT_MIN_SAMPLES", "8"))
ONLINE_RELIABILITY_PRIOR_SAMPLES = float(os.getenv("TE_BALANCEAI_RELIABILITY_PRIOR_SAMPLES", "24"))

# Candidate extra features that may be admitted to a V6 challenger once enough
# exact online snapshots contain them. The retrainer checks coverage first.
ONLINE_EXTRA_FEATURE_PREFIXES = (
    "future__weather_", "future__entsoe_", "future__mcp_", "future__revision__",
    "lag1q__fi_surplus_deficit_rt", "lag1q__fi_electric_boiler_rt",
    "lag1q__fi_frequency_rt", "lag1q__fi_wind_generation_rt",
    "lag1q__fi_hydro_generation_rt", "lag1q__fi_nuclear_generation_rt",
    "lag4q__fi_surplus_deficit_rt", "lag4q__fi_electric_boiler_rt",
    "lag4q__fi_frequency_rt", "lag4q__fi_wind_generation_rt",
    "lag4q__fi_hydro_generation_rt", "lag4q__fi_nuclear_generation_rt",
    "weather_wind_ramp_3h", "weather_temperature_ramp_3h",
    "weather_error_", "reserve_stress_index", "reserve_stress_abs",
    "production_plan_error_proxy", "live_chp_total", "live_bess_discharge", "live_shortage_status",
    "mcp_", "live_power_system_state", "live_net_import_export",
    "live_north_south_flow", "live_cumulative_balance", "live_wind_capacity_factor", "live_wind_forecast_capacity_factor",
    "online_", "live_generation_mix_renewable_share", "live_system_stress_proxy",
)

# ---------------------------------------------------------------------------
# V7 performance visibility + daily self-learning checkpoint.
# Hourly adaptation continues as before. Once each Helsinki day, V7 writes a
# performance report and conditionally runs champion/challenger retraining.
# ---------------------------------------------------------------------------
ONLINE_DAILY_LEARNING_LOCAL_HOUR = int(os.getenv("TE_BALANCEAI_DAILY_LEARNING_LOCAL_HOUR", "14"))
ONLINE_DAILY_LEARNING_LOCAL_MINUTE = int(os.getenv("TE_BALANCEAI_DAILY_LEARNING_LOCAL_MINUTE", "0"))
ONLINE_DAILY_MIN_EVALUATIONS = int(os.getenv("TE_BALANCEAI_DAILY_MIN_EVALUATIONS", "32"))
ONLINE_DAILY_REPORT_DIR = ONLINE_REPORTS_DIR / "daily"
