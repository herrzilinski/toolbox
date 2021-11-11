import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import chi2
from statsmodels.tsa.stattools import adfuller, kpss
from scipy import signal
import time


def Rolling_Mean_Var(data, dataname=None):
    mean = []
    var = []
    for j in range(len(data)):
        mean.append(np.mean(data[:j+1]))
        var.append(np.var(data[:j+1]))
    fig, axs = plt.subplots(2)
    if dataname is None:
        fig.suptitle(f'Rolling Mean & Variance Plot')
    else:
        fig.suptitle(f'Rolling Mean & Variance Plot of {dataname}')
    axs[0].plot(mean)
    axs[0].set(ylabel='Mean')
    axs[0].grid()
    axs[1].plot(var, 'tab:orange')
    axs[1].set(ylabel='Variance')
    axs[1].grid()
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


def ACF_parameter(series, lag=None, removeNA=False, two_sided=False):
    T = len(series)
    if removeNA:
        series = [x for x in series if np.isnan(x) == False]
    else:
        series = list(series)
    if lag is None:
        lag = min(int(10 * np.log10(T)), T - 1)
    res = []
    for i in np.arange(0, lag + 1):
        res.append(AutoCorrelation(series, i))
    if two_sided:
        res = np.concatenate((np.reshape(res[::-1], lag + 1), res[1:]))
    else:
        res = np.array(res)
    return res


def ACF_Plot(series, lag=None, ax=None, plt_kwargs={}, removeNA=False):
    T = len(series)
    if removeNA:
        series = [x for x in series if np.isnan(x) == False]
    else:
        series = list(series)
    if lag is None:
        lag = min(int(10 * np.log10(T)), T - 1)
    if ax is None:
        ax = plt.gca()
    ax.stem(np.linspace(-lag, lag, 2 * lag + 1), ACF_parameter(series, lag, two_sided=True), markerfmt='ro', **plt_kwargs)
    m = 1.96/np.sqrt(2*len(series)-1)
    ax.axhspan(-m, m, alpha=0.2, color='blue')
    ax.set(xlabel='Lag', ylabel='Auto-Correlation')
    return ax


def GPAC_cal(series, lags, Lj, Lk, cmap='RdBu', ry_2=None):
    if ry_2 is not None:
        if not np.array_equal(ry_2, ry_2[::-1]):
            ry_2 = np.concatenate((np.reshape(ry_2[::-1], len(ry_2)), ry_2[1:]))
    else:
        ry_2 = ACF_parameter(series, lags, two_sided=True)
    center = int((len(ry_2)-1)/2)
    if min(Lk, Lj) <= 3:
        raise Exception('Length of the table is recommended to be at least 4')
    table = []
    for j in range(Lj):
        newrow = []
        for k in range(1, Lk + 1):
            num = np.array([]).reshape(k, 0)
            for p in range(k):
                if p != k - 1:
                    newcol = []
                    for q in range(k):
                        newcol.append([ry_2[center + j + q - p]])
                    num = np.hstack((num, newcol))
                else:
                    newcol = []
                    for q in range(k):
                        newcol.append([ry_2[center + 1 + j + q]])
                    num = np.hstack((num, newcol))

            den = np.array([]).reshape(k, 0)
            for p in range(k):
                newcol = []
                for q in range(k):
                    newcol.append([ry_2[center + j + q - p]])
                den = np.hstack((den, newcol))

            # Cramer's Rule
            phi = np.round(np.linalg.det(num) / np.linalg.det(den), 3)
            newrow.append(phi)
        table.append(newrow)

    table = pd.DataFrame(table)
    table.columns = [str(x) for x in range(1, Lk + 1)]

    sns.heatmap(table, annot=True, vmin=-1, vmax=1, cmap=cmap)
    plt.title(f'Generalized Partial AutoCorrelation Table')
    plt.tight_layout()
    plt.show()



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
    if m % 2 == 1:
        k = int((m-1)/2)
        res.extend(np.zeros(k) + np.nan)
        for i in range(len(series)-2*k):
            new = np.average(series[i:i+m])
            res.append(new)
    else:
        if m2 is None:
            m2 = int(input('Please provide the second MA order'))
            if m2 % 2 == 1 or m2 <= 0:
                raise Exception('Second MA order must be even number larger than 0.')
        temp = []
        k = int(m/2-1)
        temp.extend(np.zeros(k) + np.nan)
        for i in range(len(series)-m+1):
            new = np.average(series[i:i+m])
            temp.append(new)
        res.extend(np.zeros(k+1) + np.nan)
        for i in range(len(temp)-m2):
            new = np.average(temp[i+k:i+k+m2])
            res.append(new)
    if type(series) == pd.Series:
        res = pd.Series(res, index=series.index[:len(res)])
    return res


class AR_process:
    def generate(self, param=None, size=None):
        if size is None:
            size = int(input('Please enter the number of samples'))

        if param is None:
            order = int(input('Please enter the order of AR process'))
            param = [float(x) for x in input('Please enter the corresponding parameters of AR process').split(',')]
            if order != len(param):
                raise Exception('Number of parameter does not match with AR order given.')
        else:
            order = int(len(param))

        e = np.random.normal(0, 1, size)
        nom = [1]
        nom.extend(np.zeros(order))
        den = [1]
        den.extend(float(z) for z in param)
        system = (nom, den, 1)
        t, y_res = signal.dlsim(system, e)

        return y_res

    def param_estimation(self, data, order):
        if type(data) == np.ndarray:
            data = pd.Series(data.flat)
        inter = list(np.zeros(order))
        inter += data.tolist()

        X = []
        for i in range(len(data)):
            new = []
            for j in range(order):
                new.append(inter[i+j])
            X.append(new)

        X = pd.DataFrame(X)
        X = X.loc[:, ::-1]
        beta = np.linalg.inv(X.T @ X) @ X.T @ data

        return beta


class MA_process:
    def generate(self, param=None, size=None):
        if size is None:
            size = int(input('Please enter the number of samples'))

        if param is None:
            order = int(input('Please enter the order of MA process'))
            param = [float(x) for x in input('Please enter the corresponding parameters of MA process').split(',')]
            if order != len(param):
                raise Exception('Number of parameter does not match with MA order given.')
        else:
            order = int(len(param))

        e = np.random.normal(0, 1, size)
        nom = [1]
        nom.extend(float(z) for z in param)
        den = [1]
        den.extend(np.zeros(order))
        system = (nom, den, 1)
        t, y_res = signal.dlsim(system, e)
        y_res = pd.Series(y_res.flat)

        return y_res


class ARMA_Generate:

    def __init__(self, mean_e=None, var_e=None, size=None, ARparam=None, MAparam=None, arma_process=None, ry_2=None):
        self.mean_e = mean_e
        self.var_e = var_e
        self.size = size
        self.ARparam = ARparam
        self.MAparam = MAparam
        self.arma_process = arma_process
        self.ry_2 = ry_2

    def samples(self):
        if self.size is None:
            self.size = int(input('Enter the number of data samples:'))
        if self.mean_e is None:
            self.mean_e = float(input('Enter the mean of white noise:'))
        if self.var_e is None:
            self.var_e = float(input('Enter the variance of white noise:'))
        if self.ARparam is None:
            self.ARparam = [float(x) for x in input('Enter the coefficients of AR:\n'
                                                    'e.g. 0.5, 0.2').split(',')]
        if self.MAparam is None:
            self.MAparam = [float(x) for x in input('Enter the coefficients of MA:\n'
                                                    'e.g. 0.5, -0.4').split(',')]

        params = [self.ARparam, self.MAparam]
        maxorder = max(len(self.ARparam), len(self.MAparam))

        for i in range(len(params)):
            if len(params[i]) < maxorder:
                params[i] = np.r_[params[i], np.zeros(maxorder - len(params[i]))]

        self.ARparam, self.MAparam = params
        ar = np.r_[1, self.ARparam]
        ma = np.r_[1, self.MAparam]
        self.arma_process = sm.tsa.ArmaProcess(ar, ma)
        mean_y = self.mean_e * (1 + np.sum(self.MAparam)) / (1 + np.sum(self.ARparam))
        y = self.arma_process.generate_sample(self.size, np.sqrt(self.var_e) + mean_y)
        return y

    def thoretcial_ACF(self, lag, arma_process=None, two_sided=False):
        if arma_process is not None:
            ry = arma_process.acf(lags=lag+1)
        else:
            ry = self.arma_process.acf(lags=lag+1)
        if two_sided:
            self.ry_2 = self.two_sided(ry)
            return self.ry_2
        else:
            return ry

    def two_sided(self, ry):
        ry_2 = np.concatenate((np.reshape(ry[::-1], len(ry)), ry[1:]))
        return ry_2


class ARMA_Estimate:
    def __init__(self, series, na, nb, maxiter=100, mu=0.01, max_mu=10000, d=10**(-6)):
        self.series = series
        self.na = na
        self.nb = nb
        self.maxiter = maxiter
        self.mu = mu
        self.max_mu = max_mu
        self.d = d
        self.y_hat = None
        self.e_hat = None
        self.theta_hat = None
        self.cov_theta = None
        self.var_e = None
        self.SSE_collect = []

    def e_sse_cal(self, series, theta, na, nb):
        nom = np.r_[1, np.zeros(max(na, nb))]
        den = np.r_[1, np.zeros(max(na, nb))]
        nom[1:na + 1] = theta[:na]
        den[1:nb + 1] = theta[na:]
        system = (nom, den, 1)
        _, e = signal.dlsim(system, series)
        e = np.reshape(e, [len(e), ])
        SSE = e @ e.T
        return e, SSE

    def parameters(self, debug_info=False):
        if debug_info:
            start_time = time.time()
        n = self.na + self.nb
        theta = np.zeros(n)

        for j in range(self.maxiter):
            if j == self.maxiter:
                print('Could not complete before reaching maximum iterations')
                break

            else:
                e, SSE = self.e_sse_cal(self.series, theta, self.na, self.nb)
                X = []
                for i in range(n):
                    theta_d = theta.copy()
                    theta_d[i] = theta_d[i] + self.d
                    e_d, _ = self.e_sse_cal(self.series, theta_d, self.na, self.nb)
                    x = (e - e_d) / self.d
                    X.append(x)
                X = np.matrix(X).T
                A = X.T @ X
                g = (X.T @ e).T
                delta_theta = np.linalg.inv(A + self.mu * np.identity(n)) @ g
                delta_theta = delta_theta.A1
                new_theta = delta_theta + theta
                new_e, new_SSE = self.e_sse_cal(self.series, new_theta, self.na, self.nb)

                if j == 0:
                    self.SSE_collect.append(SSE)
                else:
                    self.SSE_collect.append(new_SSE)

                if new_SSE < SSE:
                    if max(delta_theta) < self.d * 100:
                        self.e_hat = new_e
                        self.y_hat = self.series - self.e_hat
                        self.theta_hat = new_theta
                        self.var_e = new_SSE / (len(e) - n)
                        self.cov_theta = self.var_e * np.linalg.inv(A)
                        break
                    else:
                        theta = new_theta
                        self.mu = self.mu / 10
                else:
                    self.mu = self.mu * 10
                    if self.mu > self.max_mu:
                        print('Could not complete before reaching maximum mu')
                        break

        if debug_info:
            print(f'Estimation finished in {len(self.SSE_collect)} iterations in {time.time() - start_time} seconds')
            print(self.SSE_collect)

        return self.theta_hat

    def plot_prediction(self):
        fig, ax = plt.subplots()
        ax.plot(self.series, label='Training Data')
        ax.plot(self.y_hat, label='Predictions')
        fig.suptitle('One Step Ahead Predictions')
        ax.set(xlabel='# of samples', ylabel='value')
        ax.legend()
        fig.show()

    def plot_SSE(self):
        plt.plot(np.arange(1, len(self.SSE_collect) + 1, 1), self.SSE_collect)
        plt.title('SSE Curve')
        plt.xlabel('# of iterations')
        plt.xticks(np.arange(1, len(self.SSE_collect) + 1, 1))
        plt.ylabel('SSE')
        plt.grid()
        plt.tight_layout()
        plt.show()

    def confidence_interval(self):
        for i in range(len(self.theta_hat)):
            upper = self.theta_hat[i] + 2 * np.sqrt(self.cov_theta[i, i])
            lower = self.theta_hat[i] - 2 * np.sqrt(self.cov_theta[i, i])
            print(f'The C.I. of parameter{i + 1} is {lower:.5f} to {upper:.5f}')

    def residual_whiteness(self, lags, alpha=0.01):
        DOF = lags - self.na - self.nb
        re = ACF_parameter(self.e_hat, lags, two_sided=False)
        Q = len(self.e_hat) * np.sum((re[1:]) ** 2)
        chi_crit = chi2.ppf(1 - alpha, DOF)
        print(f'Chi\u00b2 test Q value of residual is {Q}.')
        print(f'Critical value under alpha={alpha*100}% is {chi_crit}')
        print(f'It is {Q<chi_crit} that the residual is white noise.')


def whiteness_test(e, lags, dof):
    re = ACF_parameter(e, lags, two_sided=False)
    Q = len(e) * np.sum((re[1:]) ** 2)
    alpha = 0.01
    chi_crit = chi2.ppf(1 - alpha, dof)
    print(f'Chi\u00b2 test Q value of residual is {Q}.')
    print(f'Critical value under alpha={alpha * 100}% is {chi_crit}')
    print(f'It is {Q < chi_crit} that the residual is white noise.')


def ACF_PACF_Plot(series, lags, series_name=None):
    fig, axs = plt.subplots(2, 1)
    sm.graphics.tsa.plot_acf(series, lags, ax=axs[0])
    sm.graphics.tsa.plot_pacf(series, lags, ax=axs[1])
    if series_name is None:
        axs[0].set(title=f'ACF Plot of {series_name}')
        axs[1].set(title=f'PACF Plot of {series_name}')
    else:
        axs[0].set(title='ACF Plot')
        axs[1].set(title='PACF Plot')
    fig.tight_layout()
    fig.show()
