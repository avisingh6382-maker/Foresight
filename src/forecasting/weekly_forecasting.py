
import numpy as np
import pandas as pd


def forecast_next_week(weekly_history, window=7):
    """
    Forecast next week's spending using rolling median.

    Parameters
    ----------
    weekly_history : pandas.Series
        Historical weekly spending values in INR.

    window : int
        Number of recent weeks used for forecasting.

    Returns
    -------
    float
        Forecasted next-week spending in INR.
    """

    if not isinstance(weekly_history, pd.Series):
        raise TypeError("weekly_history must be a pandas Series.")

    if window < 1:
        raise ValueError("window must be at least 1.")

    history = pd.to_numeric(
        weekly_history, errors="coerce"
    ).dropna()

    if len(history) < window:
        raise ValueError(
            f"At least {window} valid weeks are required."
        )

    if not np.isfinite(history.to_numpy()).all():
        raise ValueError("History contains invalid values.")

    if (history < 0).any():
        raise ValueError("Spending values cannot be negative.")

    forecast = history.iloc[-window:].median()

    return max(0.0, float(forecast))


def forecast_with_bundle(weekly_history, bundle):
    """
    Forecast using a saved ForeSight configuration.
    """

    window = bundle["primary_window"]

    return forecast_next_week(
        weekly_history,
        window=window
    )
