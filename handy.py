import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss


def Rolling_Mean_Var(data):
    for i in data.columns:
        mean = []
        var = []
        for j in range(len(data[i])):
            mean.append(np.mean(data[i].head(j + 1)))
            var.append(np.var(data[i].head(j + 1)))
        fig, axs = plt.subplots(2)
        fig.suptitle(f'Rolling Mean & Variance Plot of {i}')
        plt.grid()
        axs[0].plot(mean)
        axs[0].set(ylabel='Mean')
        axs[1].plot(var, 'tab:orange')
        axs[1].set(ylabel='Variance')
        plt.xlabel('# of Samples')
        fig.show()


def ADF_Cal(x):
    result = adfuller(x)

    print("ADF Statistic: %f" % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))


# The above section of code is provided by Professor R.Jafari


def kpss_test(timeseries):
    print('Results of KPSS Test:')
    kpsstest = kpss(timeseries, regression='c', nlags="auto")
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic', 'p-value', 'Lags Used'])
    for key, value in kpsstest[3].items():
        kpss_output['Critical Value (%s)' % key] = value
    print(kpss_output)


# The above section of code is provided by Professor R.Jafari


def AutoCorrelation(y, tau):
    # y = [x for x in y if np.isnan(x) == False]
    y_bar = np.mean(y)
    T = len(y)
    nom = np.sum((y[tau:] - y_bar) * (y[:T - tau] - y_bar))
    dnom = np.sum((y - y_bar) ** 2)
    r = nom / dnom
    return r


def ACF_parameter(series, lag=None, removeNa=False):
    if removeNa:
        series = [x for x in series if np.isnan(x) == False]
    else:
        series = list(series)
    if lag is None:
        lag = len(series) - 1
    res = []
    for i in np.abs(np.arange(-lag, lag + 1)):
        res.append(AutoCorrelation(series, i))
    return res


def ACF_Plot(series, lag=None, ax=None, plt_kwargs={}, removeNa=False):
    if removeNa:
        series = [x for x in series if np.isnan(x) == False]
    else:
        series = list(series)
    if lag is None:
        lag = len(series) - 1
    if ax is None:
        ax = plt.gca()
    ax.stem(np.linspace(-lag, lag, 2 * lag + 1), ACF_parameter(series, lag), markerfmt='ro', **plt_kwargs)
    m = 1.96/np.sqrt(2*len(series)-1)
    ax.axhspan(-m, m, alpha=0.2, color='blue')
    ax.set(xlabel='Lag', ylabel='Auto-Correlation')
    return ax


def backward_selection(y, x, maxp=0.05):
    feature = list(x.columns)
    while True:
        updated = False
        lm = sm.OLS(y, x[feature]).fit()
        p_val = lm.pvalues
        if p_val.max() > maxp:
            updated = True
            dropped = p_val.idxmax()
            feature.remove(dropped)
            print(f'Dropped feature {dropped} with p-value {p_val.max():.4f}')
        if not updated:
            break
    return lm


def MovingAverage_Cal(series, m=None, m2=None):
    if m is None:
        m = int(input('Please provide the order of MA'))
    if m == 1 and m2 == 2:
        raise Exception('MA order of 1, 2 is not acceptable.')
    res = []
    if m%2 == 1:
        k = int((m-1)/2)
        res.extend(np.zeros(k) + np.nan)
        for i in range(len(series)-2*k):
            new = np.average(series[i:i+m])
            res.append(new)
    else:
        if m2 is None:
            m2 = int(input('Please provide the second MA order'))
            if m2%2 == 1 or m2 <= 0:
                raise Exception('Second MA order must be even number larger than 0.')
        temp = []
        k = int(m/2-1)
        temp.extend(np.zeros(k) + np.nan)
        for i in range(len(series)-m+1):
            new = np.average(series[i : i + m])
            temp.append(new)
        res.extend(np.zeros(k+1) + np.nan)
        for i in range(len(temp)-m2):
            new = np.average(temp[i+k : i + k + m2])
            res.append(new)
    if type(series) == pd.Series:
        res = pd.Series(res, index=series.index[:len(res)])
    return res



