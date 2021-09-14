import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def Rolling_Mean_Var(data):
    for i in df.columns:
        mean = []
        var = []
        for j in range(len(data[i])):
            mean.append(np.mean(data[i].head(j+1)))
            var.append(np.var(data[i].head(j+1)))
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

    print("ADF Statistic: %f" %result[0])
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
        kpss_output['Critical Value (%s)'%key] = value
    print(kpss_output)
# The above section of code is provided by Professor R.Jafari


