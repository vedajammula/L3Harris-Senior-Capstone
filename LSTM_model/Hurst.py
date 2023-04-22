import numpy as np
import pandas as pd

def get_hurst_exponent(time_series, max_lag=30):
    lags = range(2, max_lag)

    # variances of the lagged differences
    tau = [np.std(np.subtract(time_series[lag:], time_series[:-lag])) for lag in lags]

    # calculate the slope of the log plot -> the Hurst Exponent
    reg = np.polyfit(np.log(lags), np.log(tau), 1)

    return reg[0]

def get_hurst_diff(window, judgement):
    combined = window.append(judgement)

    diff = get_hurst_exponent(window) - get_hurst_exponent(combined)

    return diff