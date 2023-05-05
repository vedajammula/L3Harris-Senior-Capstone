import numpy as np
import pandas as pd

def get_hurst_exponent(df, max_lag=20):
    time_series = df['Close'].to_numpy()

    lags = range(2, max_lag)

    # variances of the lagged differences
    tau = [np.std(np.subtract(time_series[lag:], time_series[:-lag])) for lag in lags]

    # calculate the slope of the log plot -> the Hurst Exponent
    reg = np.polyfit(np.log(lags), np.log(tau), 1)

    return reg[0]





def get_hurst_diff(window, judgement):
    combined = pd.concat([window,judgement])

    diff = get_hurst_exponent(window) - get_hurst_exponent(combined)
 

    if diff >= 0.5:
        return "red"
    elif diff >= 0.005:
        return "yellow"
    else:
        return "green"