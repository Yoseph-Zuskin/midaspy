################################### Credit ###################################
# Much of this file involves modifications to code written by Michael W. Mull,
# 2017 (https://github.com/mikemull/midaspy/)
##############################################################################

import numpy as np
import pandas as pd
from collections import abc
from scipy.special import gamma

def polynomial_weights(poly):
    """
    Initiate beta polynomial or exponential Almon polynomial exogenous weighting.
    
    Parameters
    ----------
    poly : str
        'exp_almon', 'beta', or 'hyperbolic' polynomial weight method.
        
    Returns
    -------
    poly_class : midas.iolib.WeightMethod
    """
    poly_class = {
        'beta': BetaWeights(1., 5.),
        'exp_almon': ExpAlmonWeights(-1., 0.),
        'hyperbolic': HyperbolicWeights(.1)
                 }
    return poly_class[poly]


class WeightMethod(object):
    """
    Weight method instance class.
    """
    def __init__(self):
        pass

    def weights(self):
        pass


class BetaWeights(WeightMethod):
    def __init__(self, theta1, theta2, theta3=None):
        self.theta1 = theta1
        self.theta2 = theta2
        self.theta3 = theta3

    def weights(self, nlags):
        """ 
        Evenly-spaced Beta polynomial weighting method.
        
        Parameters
        ----------
        nlags : int
            Number of lag terms in projection matrix.
        
        Returns
        -------
        array : numpy.array
            Polynomial weights.
        """
        eps = np.spacing(1)
        u = np.linspace(eps, 1.0 - eps, nlags)

        beta_vals = u ** (self.theta1 - 1) * (1 - u) ** (self.theta2 - 1)

        beta_vals = beta_vals / sum(beta_vals)

        if self.theta3 is not None:
            w = beta_vals + self.theta3
            return w / sum(w)

        return beta_vals

    def x_weighted(self, x, params):
        self.theta1, self.theta2 = params

        w = self.weights(x.shape[1])

        return np.dot(x, w), np.tile(w.T, (x.shape[1], 1))

    @property
    def num_params(self):
        return 2 if self.theta3 is None else 3

    @staticmethod
    def init_params():
        return np.array([1., 5.])

    
class ExpAlmonWeights(WeightMethod):
    def __init__(self, theta1, theta2):
        self.theta1 = theta1
        self.theta2 = theta2

    def weights(self, nlags):
        """
        Exponential Almon polynomial weighting method.

        Parameters
        ----------
        nlags : int
            Number of lag terms in projection matrix.
        
        Returns
        -------
        array : numpy.array
            Polynomial weights.
        """
        ilag = np.arange(1, nlags + 1)
        z = np.exp(self.theta1 * ilag + self.theta2 * ilag ** 2)
        return z / sum(z)

    def x_weighted(self, x, params):
        self.theta1, self.theta2 = params

        w = self.weights(x.shape[1])

        return np.dot(x, w), np.tile(w.T, (x.shape[1], 1))

    @property
    def num_params(self):
        return 2

    @staticmethod
    def init_params():
        return np.array([-1., 0.])

    
class HyperbolicWeights(WeightMethod):
    def __init__(self, theta):
        self.theta = theta

    def weights(self, nlags):
        """
        Exponential Almon polynomial weighting method.

        Parameters
        ----------
        nlags : int
            Number of lag terms in projection matrix.
        
        Returns
        -------
        array : numpy.array
            Polynomial weights.
        """
        u = np.arange(1, nlags + 1)
        g = gamma(u + self.theta) / (gamma(u + 1) * gamma(self.theta))
        return g / sum(g)

    def x_weighted(self, x, param):
        self.theta = param

        w = self.weights(x.shape[1])

        return np.dot(x, w), np.tile(w.T, (x.shape[1], 1))

    @property
    def num_params(self):
        return 1

    @staticmethod
    def init_params():
        return np.array([.1])

    
def jacobian_wx(x, params, weight_method):
    """
    Compute Jacobian of weighted exognenous variable.
    
    Parameters:
    x : pandas.DataFrame
        Exogenous variable lagged projection matrix.
    params : float or numpy.array
        Polynomial weighting parameter(s).
    weight_method : midaspy.iolib.WeightMethod
        Polynomial weighting instance based on spciefied method.
        
    Returns
    -------
    jacobian : numpy.ndarray
        Jacobian matrix
    """
    eps = 1e-6
    jt = []
    for i, p in enumerate(params):
        dp = np.concatenate([params[0:i], [p + eps / 2], params[i + 1:]])
        dm = np.concatenate([params[0:i], [p - eps / 2], params[i + 1:]])
        jtp, _ = weight_method.x_weighted(x, dp)
        jtm, _ = weight_method.x_weighted(x, dm)
        jt.append((jtp - jtm) / eps)

    return np.column_stack(jt)
    

def ssr(a, x, y, yl, weight_methods):
    """
    Compute errors of the MIDAS equation
    
    Parameters
    ----------
    a : numpy.array
        Regression coefficients and weighting parameters.
    x : dictionary of pandas.DataFrames
        All exogenous variables' higher-to-lower frequency projection matrices.
    y : pandas.DataFrame
        training set of endogenous target data.
    yl : pandas.DataFrame
        Autoregressive distributed endogenous lag terms.
    weight_methods : dictionary of strings
        Specified polynomial weighting method for each exogenous projection matrix.

    Returns
    -------
    error : numpy.array
        Error values for each predicted value relative to actual value.
    """
    exog_vars = list(x.keys())
    alpha, betas, thetas, xws, num_exog = a[0], {}, {}, {}, len(exog_vars)
    error = y.values - alpha
    for var, i in zip(exog_vars, range(1, num_exog + 1)):
        betas[var] = a[i]
    num_params = [polynomial_weights(poly).num_params for poly in weight_methods.values()]
    thetas_order = np.cumsum(num_params) - num_params + num_exog + 1
    for var, i in zip(exog_vars, thetas_order):
        thetas[var] = a[i:i + polynomial_weights(weight_methods[var]).num_params]
    for var in exog_vars:
        weight_method = polynomial_weights(weight_methods[var])
        xw, w = weight_method.x_weighted(x[var], thetas[var])
        xws[var] = xw.reshape((len(xw), 1))
    for var in exog_vars:
        error -= betas[var] * xws[var]
    if yl is not None:
        error = error - np.dot(yl, a[-1 * yl.shape[1]:].reshape(a[-1 * yl.shape[1]:].shape[0], 1 if yl.shape[1] is not None else None))
    return error.reshape((len(error),))

def jacobian(a, x, y, yl, weight_methods):
    """
    Compute Jacobian of the MIDAS equation

    Parameters
    ----------
    a : numpy.array
        Regression coefficients and weighting parameters.
    x : dictionary of pandas.DataFrames
        All exogenous variables' higher-to-lower frequency projection matrices.
    y : pandas.DataFrame
        training set of endogenous target data.
    yl : pandas.DataFrame
        Autoregressive distributed endogenous lag terms.
    weight_methods : dictionary of strings
        Specified polynomial weighting method for each exogenous projection matrix.

    Returns
    -------
    jac_e : numpy.ndarray
        Jacobian matrix.
    """
    exog_vars = list(x.keys())
    alpha, betas, thetas, xws, jwxs, num_exog = a[0], {}, {}, {}, {}, len(exog_vars)
    for var, i in zip(exog_vars, range(1, num_exog + 1)):
        betas[var] = a[i]
    num_params = [polynomial_weights(poly).num_params for poly in weight_methods.values()]
    thetas_order = np.cumsum(num_params) - num_params + num_exog + 1
    for var, i in zip(exog_vars, thetas_order):
        thetas[var] = a[i:i + polynomial_weights(weight_methods[var]).num_params]
    for var in exog_vars:
        weight_method = polynomial_weights(weight_methods[var])
        xw, w = weight_method.x_weighted(x[var], thetas[var])
        xws[var], jwxs[var] = xw.reshape((len(xw), 1)), jacobian_wx(x[var], thetas[var], polynomial_weights(weight_methods[var]))
    jac_e = [np.ones((len(y), 1))]
    for var in exog_vars:
        jac_e.append(xws[var])
    for var in exog_vars:
        jac_e.append(betas[var] * jwxs[var])
    if yl is not None:
        jac_e.append(yl)
    jac_e = -1 * np.concatenate(jac_e, axis = 1)
    return jac_e

def low_freq_projection(x, xlag, horizon, target_dates):
    """
    Project high frequency variable onto lower frequency of regression
    target through a lagged matrix
    
    Parameters
    ----------
    x : pandas.Series
        Exogenous variable data with datetimes index.
    xlag : intereger
        Number of lagged observations to include in the projection matrix
    horizon : integer
        Number of high-frequency observations prior to each target date.
        For example, with horizon of 1 and target date of 2012-12-31, the
        first lagged term in the projection matrix row corresponding to this
        date will be 2012-12-30, then 2012-12-29, and so on until there are
        xlag terms in the projection matrix.
    target_dates : pandas.core.indexes.datetimes.DatetimeIndex
        Low-frequency target dates. Used to select only the rows of the
        projection matrix which correspond to date of target values.
        
    Returns
    -------
    projection_matrix : pandas.DataFrame
        Lagged projection matrix of x containing xlag terms.
    """
    projection_matrix = pd.DataFrame()
    for lag in range(horizon, xlag + horizon):
        projection_matrix = pd.concat([projection_matrix, x.shift(lag).rename(x.name+' t-{}'.format(lag))], axis=1)
    projection_matrix = projection_matrix.loc[target_dates]
    return projection_matrix

def nested_dict_iter(nested):
    """
    Yield nested dictionary iterator from dictionary of dictionaries.
    
    Author: Jonathan Scott Enderle
    Source: https://stackoverflow.com/questions/10756427/
    
    Parameter
    nested : dictionary
        Dictionary of dictionaries.
        
    Returns
    -------
    generator
        Iterable object containing nested key and value pairs.
    """
    for key, value in nested.items():
        if isinstance(value, abc.Mapping):
            yield from nested_dict_iter(value)
        else:
            yield key, value