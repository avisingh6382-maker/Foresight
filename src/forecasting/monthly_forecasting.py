
import numpy as np
import pandas as pd

from numbers import Integral


def forecast_monthly_spending(
    monthly_spending,
    periods=1
):
    """
    Forecast monthly spending using the Naive benchmark.

    Parameters
    ----------
    monthly_spending : pandas.Series
        Historical monthly spending values.
        Index must contain monthly Period values or dates.

    periods : int
        Number of future months to forecast.
        Must be a positive integer.

    Returns
    -------
    pandas.DataFrame
        Forecast values indexed by future monthly Period values.

    Notes
    -----
    Naive forecasting repeats the latest observed monthly
    spending value for each future month.

    This is a baseline method, not a model that learns
    spending behaviour or predicts unusual events.
    """

    # --------------------------------------------------------
    # 1. VALIDATE INPUT TYPE
    # --------------------------------------------------------

    if not isinstance(monthly_spending, pd.Series):
        raise TypeError(
            "monthly_spending must be a pandas Series."
        )

    # --------------------------------------------------------
    # 2. VALIDATE FORECAST HORIZON
    # --------------------------------------------------------

    if (
        isinstance(periods, bool)
        or not isinstance(periods, Integral)
        or periods < 1
    ):
        raise ValueError(
            "periods must be a positive integer."
        )

    # --------------------------------------------------------
    # 3. VALIDATE INDEX TYPE
    # --------------------------------------------------------

    if not isinstance(
        monthly_spending.index,
        (pd.PeriodIndex, pd.DatetimeIndex)
    ):
        raise TypeError(
            "Index must be a pandas PeriodIndex "
            "or DatetimeIndex."
        )

    if len(monthly_spending) == 0:
        raise ValueError(
            "Historical spending cannot be empty."
        )

    # --------------------------------------------------------
    # 4. CONVERT INDEX TO MONTHLY PERIODS
    # --------------------------------------------------------

    historical = monthly_spending.copy()

    try:

        if isinstance(historical.index, pd.PeriodIndex):

            historical.index = historical.index.asfreq("M")

        else:

            # Remove timezone while preserving local dates.
            if historical.index.tz is not None:
                historical.index = (
                    historical.index.tz_localize(None)
                )

            historical.index = historical.index.to_period("M")

    except Exception as error:

        raise ValueError(
            "Unable to interpret historical dates as months."
        ) from error

    # --------------------------------------------------------
    # 5. CHECK DUPLICATE MONTHS
    # --------------------------------------------------------

    if historical.index.has_duplicates:
        raise ValueError(
            "Historical data contains duplicate months."
        )

    # --------------------------------------------------------
    # 6. CONVERT SPENDING VALUES TO NUMERIC
    # --------------------------------------------------------

    numeric_history = pd.to_numeric(
        historical,
        errors="coerce"
    )

    invalid_values = (
        historical.notna()
        & numeric_history.isna()
    )

    if invalid_values.any():
        raise ValueError(
            "Historical spending contains non-numeric values."
        )

    # --------------------------------------------------------
    # 7. REJECT INFINITE VALUES
    # --------------------------------------------------------

    non_missing_values = numeric_history.dropna()

    if not np.isfinite(
        non_missing_values.to_numpy(dtype=float)
    ).all():

        raise ValueError(
            "Historical spending must contain only finite values."
        )

    # --------------------------------------------------------
    # 8. REJECT NEGATIVE SPENDING ANYWHERE IN HISTORY
    # --------------------------------------------------------

    if (non_missing_values < 0).any():
        raise ValueError(
            "Historical spending cannot contain negative values."
        )

    # --------------------------------------------------------
    # 9. REMOVE MISSING OBSERVATIONS AND SORT
    # --------------------------------------------------------

    historical = numeric_history.dropna().sort_index()

    if historical.empty:
        raise ValueError(
            "No valid historical spending observations."
        )

    # --------------------------------------------------------
    # 10. GET LATEST OBSERVED SPENDING
    # --------------------------------------------------------

    last_value = float(historical.iloc[-1])

    last_index = historical.index[-1]

    # --------------------------------------------------------
    # 11. GENERATE FUTURE MONTHLY INDEX
    # --------------------------------------------------------

    future_index = pd.period_range(
        start=last_index + 1,
        periods=int(periods),
        freq="M"
    )

    # --------------------------------------------------------
    # 12. APPLY NAIVE FORECASTING
    # --------------------------------------------------------

    forecast = pd.DataFrame(
        {
            "Forecast_Spending": [
                last_value
            ] * int(periods)
        },
        index=future_index
    )

    forecast.index.name = "Month"

    # --------------------------------------------------------
    # 13. RETURN FORECAST
    # --------------------------------------------------------

    return forecast
