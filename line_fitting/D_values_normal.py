import os
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
TIMESERIES_PATH = os.path.join(RESULTS_DIR, "line_fitting_timeseries.csv")

# Outputs
WINDOWED_OUT_PATH = os.path.join(RESULTS_DIR, "normal_trend_score_windowed_results.csv")
MAX_OUT_PATH = os.path.join(RESULTS_DIR, "normal_trend_score_max_per_trial.csv")
THRESH_OUT_PATH = os.path.join(RESULTS_DIR, "normal_trend_score_thresholds.csv")

WINDOW = 10
STEP = 1
PERCENTILE = 95

# Local x-axis for a window of length 10: 0..9
_S = np.arange(WINDOW, dtype=float)
_S_BAR = _S.mean()
_S_DEV = _S - _S_BAR
_SS_S = np.sum(_S_DEV ** 2)


def windowed_d(z_values):
    """
    Compute windowed D(t) for one run.

    Parameters
    ----------
    z_values : array-like
        rolling_z_w10 values for one run. May contain NaNs at the start.

    Returns
    -------
    list of dict
        Each dict contains window_start, beta_local, D.
    """
    n = len(z_values)
    results = []

    for start in range(0, n - WINDOW + 1, STEP):
        window = z_values[start:start + WINDOW]

        # Skip windows that contain NaNs
        if np.any(np.isnan(window)):
            continue

        # OLS slope beta for local index 0..9
        beta = np.sum(_S_DEV * window) / _SS_S

        # Intercept alpha (not used later, but needed for residuals)
        alpha = window.mean() - beta * _S_BAR

        fitted = alpha + beta * _S
        resid = window - fitted

        # sigma_eps^2 = sum(resid^2)/(W-2)
        sigma_eps2 = np.sum(resid ** 2) / (WINDOW - 2)

        # SE(beta) = sqrt(sigma_eps^2 / sum((s_i - s_bar)^2))
        se_beta = np.sqrt(sigma_eps2 / _SS_S) if sigma_eps2 >= 0 else np.nan

        d = abs(beta / se_beta) if (se_beta is not None and se_beta > 0) else np.nan

        results.append(
            {
                "window_start": start,
                "beta_local": beta,
                "D": d,
            }
        )

    return results


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = pd.read_csv(TIMESERIES_PATH)

    # Keep only normal runs
    ts = ts[ts["mode"] == "normal"].copy()

    # Ensure sorted order inside each run
    ts = ts.sort_values(["counter", "seed", "sample_index"])

    window_rows = []
    trial_rows = []

    for (counter, seed), group in ts.groupby(["counter", "seed"]):
        z_values = group["rolling_z_w10"].to_numpy(dtype=float)
        sample_indices = group["sample_index"].to_numpy()

        wd = windowed_d(z_values)

        if len(wd) == 0:
            trial_rows.append(
                {
                    "counter": counter,
                    "seed": seed,
                    "max_D": np.nan,
                    "num_windows": 0,
                }
            )
            continue

        # Save all window-level D values
        for item in wd:
            start = item["window_start"]
            window_end_t = sample_indices[start + WINDOW - 1]

            window_rows.append(
                {
                    "counter": counter,
                    "seed": seed,
                    "window_start": start,
                    "window_end_sample_index": window_end_t,
                    "beta_local": item["beta_local"],
                    "D": item["D"],
                }
            )

        d_vals = np.array([item["D"] for item in wd], dtype=float)
        d_vals = d_vals[np.isfinite(d_vals)]

        trial_rows.append(
            {
                "counter": counter,
                "seed": seed,
                "max_D": np.max(d_vals) if len(d_vals) > 0 else np.nan,
                "num_windows": len(d_vals),
            }
        )

    windowed_df = pd.DataFrame(window_rows)
    trial_df = pd.DataFrame(trial_rows)

    # Threshold per counter = 95th percentile of the 20 max_D values
    thresh_df = (
        trial_df.groupby("counter")["max_D"]
        .agg(
            threshold=lambda x: np.percentile(x.dropna(), PERCENTILE) if x.dropna().size > 0 else np.nan,
            mean_max_D="mean",
            std_max_D="std",
            min_max_D="min",
            max_max_D="max",
            n_trials="count",
        )
        .reset_index()
    )

    # Save outputs
    windowed_df.to_csv(WINDOWED_OUT_PATH, index=False)
    trial_df.to_csv(MAX_OUT_PATH, index=False)
    thresh_df.to_csv(THRESH_OUT_PATH, index=False)

    print(f"Saved window-level D values to: {WINDOWED_OUT_PATH}")
    print(f"Saved per-trial max D values to: {MAX_OUT_PATH}")
    print(f"Saved counter thresholds to: {THRESH_OUT_PATH}")

    print("\nPer-trial max D values:")
    print(trial_df.sort_values(["counter", "seed"]).to_string(index=False))

    print("\nThresholds per counter:")
    print(thresh_df.sort_values("counter").to_string(index=False))


if __name__ == "__main__":
    main()