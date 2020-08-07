import numpy as np, pandas as pd
from midaspy.iolib import *
from dateutil.relativedelta import relativedelta
from scipy.optimize import least_squares
from scipy.stats import t

class MIDASRegressor(object):
    """
    Mixed Data Sampling (MIDAS) Regressor Class
    
    Author: Yoseph Zuskin
    License: MIT
    
    Parameters
    ----------
    endog:
        Lower frequency endogenous target data (as pandas dataframe or
        series).
    exog:
        Higher frequency exogenous data (as pandas dataframe, series, or
        as list or tuple containing multiple dataframes or series).
    ylag:
        Number of autoregressive distributed lag terms (optional).
    xlag:
        Time horizon to be used for projecting the higher-frequency data
        onto lower-frequency lagged matrix. Can be set for all or set for
        each parameter as a list or tuple of xlag parameters (must be same
        length as number of exogenous variables and follow their order of
        appearance). This parameters is optional, if not specified, default
        parameters will be used based on the difference in frequencies.
    horizon:
        Number of high-frequency observations prior to target date. Will
        use default oo 1 if not specified. Use 0 for nowcasting. Can be
        set for all predictiors or, similarly to xlag, as a list or tuple
        of the same length as number of predictors, in which case the order
        of predictor appearance in the exog param will be used to match with
        elements of this parameter.
    poly:
        Beta, exponential Almon, or hyperbolic scheme polynomial weighting
        method. Will use default of Beta if not specified. Can be set for
        all predictiors or, similarly to xlag, as a list or tuple of the
        same length as number of predictors, in which case the order of
        predictor appearance in the exog param will be used to match with
        elements of this parameter.
    
    Attributes
    ----------
    orig_endog : pandas.DataFrame
        Original endogenous dependent values with datetimes index.
    endog_name : string
        Name of endogenous dependent variable.
    target_frequency: stirng
        Inferred time frequency of endogenous dependent variable.
    adl : pandas.DataFrame
        Autoregressive distributed lag terms based on ylag parameter.
    horizon: tuple
        Horizon parameters for each exogenous variable.
    exog: dictionary
        Data series and related properties of exogenous variables.
        Can be parsed using midaspy.iolib.nested_dict_iter to access
        the data and properties of all variables. For a specified
        variable, must specify its position in the exogenous input
        using the following example scheme: exog['df1']['var_name']
        for var_name in the first (or only) dataframe or series given
        in the exogenous input parameter. Will include coeficients
        and weighting parameters after fitting the model.
    """
    def __init__(self, endog, exog, ylag=None, xlag=None, horizon=1, poly=None):
        # check to ensure the endogenous is a series or a data frame with only one column
        if type(endog) == pd.Series or (type(endog) == pd.DataFrame and endog.shape[1] == 1):
            if endog.isnull().values.any():
                raise ValueError('endogenous data should not have any missing values')
            if type(endog.index) == pd.core.indexes.datetimes.DatetimeIndex:
                endog_datetimes_index = endog.index
                self.orig_endog = pd.DataFrame(endog)
                self.endog_name = pd.DataFrame(endog).columns[0]
            else:
                try:
                    endog_datetimes_index = pd.to_datetime(endog.index)
                    endog.index = pd.DataFrame(endog).set_index(endog_datetimes_index)
                    self.orig_endog = pd.DataFrame(endog)
                    self.endog_name = pd.DataFrame(endog).columns[0]
                except:
                    raise ValueError('endogenous data must have a datetimes index')
        elif type(endog) == pd.DataFrame and endog.shape[1] != 1:
            raise ValueError('please specificy the target column in the endogenous data frame, this class is not configured for multiviariat')
        else:
            raise ValueError('endogenous data must be given as pandas.DataFrame or pandas.Series with datetimes index')
        # determine target time frequency
        low_freq = pd.infer_freq(endog_datetimes_index)[0]
        self.target_frequency = low_freq
        # check to ensure ylag parameter is a valid value and generate autoregressive distributed lags (adl)
        if ylag is not None:
            if type(ylag) == int and ylag > 0:
                adl = pd.DataFrame()
                if type(endog) == pd.Series:
                    for lag in range(1, ylag + 1):
                        adl = pd.concat([adl, endog.shift(lag).rename(endog.name+' t-{}'.format(lag))], axis = 1)
                elif type(endog) == pd.DataFrame:
                    for lag in range(1, ylag + 1):
                        adl = pd.concat([adl, endog.shift(lag).rename(columns={endog.columns[0]:endog.columns[0]+' t-{}'.format(lag)})], axis = 1)
                adl = adl.iloc[ylag:, :]
                self.adl = adl
                if type(endog.index) == pd.core.indexes.datetimes.DatetimeIndex:
                    endog = pd.DataFrame(endog).iloc[ylag:]
                    self.orig_endog = endog
                else:
                    try:
                        endog_datetimes_index = pd.to_datetime(endog.index)
                        endog.index = pd.DataFrame(endog).set_index(endog_datetimes_index)
                        endog = pd.DataFrame(endog)[ylag:]
                        self.orig_endog = endog
                    except:
                        raise ValueError('endogenous data must be given as pandas.DataFrame or pandas.Series with datetimes index')
        else:
            self.adl = None
        # check to ensure the horizon parameter is valid
        if type(horizon) == int and horizon >= 0:
            self.horizon = horizon
        else:
            raise ValueError('horizon parameter must be an integer greater or equal to zero (default is 1 high-frequency period prior to the low-frequency target date; use 0 for nowcasting)')
        # check to ensure the exogenous data is in a series or data frame, or list/tuple of such objects
        exog_list = list()
        if type(exog) == pd.Series or type(exog) == pd.DataFrame:
            if exog.isnull().values.any():
                raise ValueError('exogenous data should not have any missing values')
            if type(exog.index) != pd.core.indexes.datetimes.DatetimeIndex:
                exog_list.append(exog)
            else:
                try:
                    exog_list.append(exog.set_index(pd.to_datetime(exog.index)))
                except:
                    raise ValueError('exogenous data must have a valid datetimes index')
        elif type(exog) == list or type(exog) == tuple:
            for exog_df, i in zip(exog, range(len(exog))):
                if exog_df.isnull().values.any():
                    raise ValueError('exogenous data should not have any missing values (missing values found in exogenous element {})'.format(i))
                df_data = {}
                if type(exog_df) == pd.Series or type(exog_df) == pd.DataFrame:
                    if type(exog_df.index) == pd.core.indexes.datetimes.DatetimeIndex:
                        df = pd.DataFrame(exog_df)
                    else:
                        try:
                            df = pd.DataFrame(exog_df.set_index(pd.to_datetime(exog_df.index)))
                        except:
                            raise ValueError('element {} in list of exogenous dataframe does not have a valid datetimes index'.format(i))
                    exog_list.append(df)
        exog_data = {}
        for exog_df, i in zip(exog_list, range(len(exog_list))):
            df_data = {}
            frequency = pd.infer_freq(exog_df.index)
            for var, j in zip(exog_df.columns, range(exog_df.shape[1])):
                df_data.update({var: {'orig': exog_df.iloc[:, j], 'freq': frequency}})
            exog_data['df{}'.format(i+1)] = df_data
         # check to ensure the polynomial weights parameter is valid
        valid_poly = ['exp_almon', 'beta', 'hyperbolic', None]
        if poly is None: 
            poly = 'beta' # use beta polynomial weights as default if not otherwise specified
        if poly in valid_poly:
            for exog_df, i in zip(exog_list, range(len(exog_list))):
                for var in exog_df.columns:
                    exog_data['df{}'.format(i+1)][var].update({
                        'poly': poly, 'num_thetas': polynomial_weights(poly).num_params
                    })
        elif type(poly) == list or type(poly) == tuple:
            if len(exog_list) == 1 and len(poly) == exog_list[0].shape[1] and all(type(w) == str or w is None for w in poly) and all(w in valid_poly for w in poly):
                for var, i in zip(exog_list[0].columns, range(len(poly))):
                    exog_data['df1'][var].update({
                        'poly': poly[i] if poly[i] is not None else 'beta',
                        'num_thetas': polynomial_weights(poly[i]).num_params
                    })
            elif all(type(p) == list or type(p) == tuple for p in poly) and all(len(p) == exog_list[i].shape[1] for p, i in zip(poly, range(len(exog_list)))):
                for exog_df, i in zip(exog_list, range(len(exog))):
                    for var, j in zip(exog_df.columns, range(len(exog_df.shape[1]))):
                        exog_data['df{}'.format(i+1)][var].update({
                            'poly': poly[i][j] if poly[i][j] is not None else 'beta',
                            'num_thetas': polynomial_weights(poly[i][j]).num_params
                        })
        else:
            raise ValueError("poly parameter must be set to 'exp_almon', 'beta', 'hyperbolic', or left as None (which will default to 'beta') to set that weight method for all variables, or a list/tuple of such parameters for each variable in the exogenous data (may enter as list of lists or tuple of tuples if multiple exogenous data frames are given)")
        # check to ensure predictor data projection time horizon is a valid value
        default = {'A': {'A': 3, 'Q': 4, 'M': 12, 'W': 52, 'D': 365, 'H': 8760, 'T': 525600, 'S': 31536000},
                   'Q': {'Q': 3, 'M': 3, 'W': 14, 'D': 98, 'H': 2190, 'T': 131400, 'S': 7884000},
                   'M': {'M': 3, 'W': 4, 'D': 30, 'H': 730, 'T': 43800, 'S': 2628000},
                   'W': {'W': 3, 'D': 7, 'H': 168, 'T': 10107, 'S': 606461},
                   'D': {'D': 3, 'H': 24, 'T': 1440, 'S': 86400},
                   'H': {'H': 3, 'T': 60, 'S': 3600},
                   'T': {'T': 3, 'S': 60},
                   'S': {'S': 3}}
        xlag_error = "xlag parameter must be integer or string (i.e. '7D' or '1W') to specify the higher-to-lower frequency projection of exogenous data"
        if xlag is None:
            for exog_df, i in zip(exog_list, range(len(exog_list))):
                for var in exog_df.columns:
                    lags = default[low_freq][exog_data['df{}'.format(i+1)][var]['freq'][0]]
                    exog_data['df{}'.format(i+1)][var].update({'lags': lags})
        elif type(xlag) == str and '.' not in xlag:
            for exog_df, i in zip(exog_list, range(len(exog_list))):
                for var in exog_df.columns:
                    if xlag[:-1].isnumeric() and type(xlag[-1]) == str:
                        lags = int(xlag[:-1]) * default[xlag[-1].capitalize()][exog_data['df{}'.format(i+1)][var]['freq'][0]]
                    elif xlag.isnumeric():
                        lags = int(xlag)
                    else:
                        raise ValueError(xlag_error)
                    exog_data['df{}'.format(i+1)][var].update({'lags': lags})
        elif type(xlag) == int:
            for exog_df, i in zip(exog_list, range(len(exog_list))):
                for var in exog_df.columns:
                    exog_data['df{}'.format(i+1)][var].update({'lags': xlag})
        elif type(xlag) == list or type(xlag) == tuple:
            if len(exog_list) == 1 and len(xlag) == exog_list[0].shape[1] and all(type(lag) in [int, str, type(None)] for lag in xlag):
                for var, i in zip(exog_list[0].columns, range(len(xlag))):
                    if type(xlag[i]) == str and xlag[i][:-1].isnumeric():
                        lags = int(xlag[i][:-1]) * default[xlag[i][-1].capitalize()][exog_data['df1'][var]['freq'][0]]
                    elif type(xlag[i]) == str and xlag[i].isnumeric():
                        lags = int(xlag[i])
                    elif type(xlag[i]) == int:
                        lags = xlag[i]
                    elif xlag[i] is None:
                        lags = default[low_freq][exog_data['df1'][var]['freq'][0]]
                    else:
                        raise ValueError(xlag_error)
                    exog_data['df1'][var].update({'lags': lags})
            elif len(exog_list) == len(xlag) and all(len(xlags) == exog_list[i].shape[1] for xlags, i in zip(xlag, range(len(exog)))):
                for exog_df, i, xlags in zip(exog_list, len(exog_list), xlag):
                    for var, j in zip(exog_df.columns, range(len(xlags))):
                        if type(xlags[j]) == str and xlags[j][:-1].isnumeric():
                            lags = int(xlags[j][:-1]) * default[xlags[j][-1].capitalize()][exog_data['df{}'.format(i+1)][var]['freq'][0]]
                        elif type(xlags[j]) == str and xlags[j].isnumeric():
                            lags = int(xlag[j])
                        elif type(xlags[j]) == int:
                            lags = xlag[j]
                        elif xlags[j] is None:
                            lags = default[low_freq][exog_data['df{}'.format(i+1)][var]['freq'][0]]
                        else:
                            raise ValueError(xlag_error)
                        exog_data['df{}'.format(i+1)][var].update({'lags': lags})
        else:
            raise ValueError(xlag_error)
        # check to ensure there is enough higher frequency data to enable
        # projection of the high-frequency variable onto the low-frequency
        for exog_df, i in zip(exog_list, range(len(exog_list))):
            for var in exog_df.columns:
                var_freq = exog_data['df{}'.format(i+1)][var]['freq'][0]
                if var_freq == 'A':
                    time_delta = relativedelta(years = horizon)
                elif var_freq == 'Q':
                    time_delta = relativedelta(months = 3 * horizon)
                elif var_freq == 'M':
                    time_delta = relativedelta(months = horizon)
                elif var_freq == 'W':
                    time_delta = relativedelta(weeks = horizon)
                elif var_freq == 'D':
                    time_delta = relativedelta(days = horizon)
                elif var_freq == 'H':
                    time_delta = relativedelta(hours = horizon)
                elif var_freq == 'T':
                    time_delta = relativedelta(minutes = horizon)
                elif var_freq == 'S':
                    time_delta = relativedelta(seconds = horizon)
                else: ### this should never be raised ###
                    raise SystemError('unknown frequency detection error, please log and report this to module author via GitHub')
                if endog_datetimes_index[-1] - time_delta not in exog_data['df{}'.format(i+1)][var]['orig'].index:
                    raise ValueError('not enough observations at the end of {} to generate last row of the lagged projection matrix'.format(var))
                available_for_row1 = exog_data['df{}'.format(i+1)][var]['orig'][:endog_datetimes_index[0]].shape[0]
                needed_for_all_rows = (exog_data['df{}'.format(i+1)][var]['lags'] + horizon)
                if available_for_row1 >= needed_for_all_rows:
                    continue
                else:
                    raise ValueError('not enough observations available for {} to create lagged projection matrix, need to have at least {} more observations'.format(
                    var, needed_for_all_rows - available_for_row1))
        # project higher frequency predictor data onto the lower frequency given specified xlag paramaeter
        for exog_df, i in zip(exog_list, range(len(exog_list))):
            for var in exog_df.columns:
                projection_matrix = {
                    'projection_matrix': low_freq_projection(exog_data['df{}'.format(i+1)][var]['orig'],
                                                             exog_data['df{}'.format(i+1)][var]['lags'],
                                                             horizon,
                                                             endog_datetimes_index[ylag if ylag is not None else 0:])
                }
                exog_data['df{}'.format(i+1)][var].update(projection_matrix)
        self.exog = exog_data
    def fit(self, bounds=(-np.inf, np.inf), method='trf', ftol=1e-09, xtol=1e-09, gtol=1e-09, x_scale=1., loss='linear',
            f_scale=1., diff_step=None, tr_solver=None, tr_options={}, jac_sparsity=None, max_nfev=5000, verbose=0):
        """
        Fit MIDAS model using linear least squares optimization
        
        Parameters
        ----------
        bounds : 2-tuple of array_like, optional
            See notes on scipy.optimize.least_squares class.
        method : {‘trf’, ‘dogbox’, ‘lm’}, optional
            See notes on scipy.optimize.least_squares class.
        ftol : float or None, optional
            See notes on scipy.optimize.least_squares class.
        xtol : float or None, optional
            See notes on scipy.optimize.least_squares class.
        gtol : float or None, optional
            See notes on scipy.optimize.least_squares class.
        x_scale : array_like or ‘jac’, optional
            See notes on scipy.optimize.least_squares class.
        loss : str or callable, optional
            See notes on scipy.optimize.least_squares class.
        f_scale : float, optional
            See notes on scipy.optimize.least_squares class.
        diff_step : None or array_like, optional
            See notes on scipy.optimize.least_squares class.
        tr_solver : {None, ‘exact’, ‘lsmr’}, optional
            See notes on scipy.optimize.least_squares class.
        tr_options : dictionary, optional
            See notes on scipy.optimize.least_squares class.
        jac_sparsity : {None, array_like, sparse matrix}, optional
            See notes on scipy.optimize.least_squares class.
        max_nfev : None or integer, optional
            See notes on scipy.optimize.least_squares class.
        verbose : {0, 1, 2}, optional
            See notes on scipy.optimize.least_squares class.
        
        Returns
        -------
        params : pd.DataFrame
            Coefficient values from model fit.
        X: pd.DataFrame
            Coefficient values from optimization results, excluding the
            polynomial weighting parameters (which can be found in exog).
        fit_model : midas.model.MIDASResultsWrapper
            Model results wrapper class.
        """
        # estimate the model coefficients using least sum of residual squares
        var_vals, exog_vars, exog_polys, xws, ws = list(), dict(), dict(), dict(), dict()
        var_vals.append(np.ones((len(self.orig_endog), 1)))
        for df in self.exog.keys():
            for var in self.exog[df].keys():
                exog_vars.update({var: self.exog[df][var]['projection_matrix']})
                exog_polys.update({var: self.exog[df][var]['poly']})
                weight_method = polynomial_weights(self.exog[df][var]['poly'])
                init_params = polynomial_weights(self.exog[df][var]['poly']).init_params()
                xws[var], ws[var] = weight_method.x_weighted(self.exog[df][var]['projection_matrix'], init_params)
                xws[var] = xws[var].reshape((len(xws[var]), 1))
                var_vals.append(xws[var])
        if self.adl is not None:
            var_vals.append(self.adl.values)
        c = np.linalg.lstsq(np.concatenate(var_vals, axis = 1), self.orig_endog, rcond = None)[0]
        param_list, num_exog = list(), len(exog_vars)
        self.num_exog = num_exog
        for p in list(c[:1 + num_exog]):
            param_list.append(p[0])
        for df in self.exog.keys():
            for var in self.exog[df].keys():
                for p in polynomial_weights(self.exog[df][var]['poly']).init_params():
                    param_list.append(p)
        if self.adl is not None:
            for p in list(c[1 + num_exog:]):
                param_list.append(p[0])
        init_params = np.array(param_list)
        fun = lambda v: ssr(v, exog_vars, self.orig_endog, self.adl, exog_polys)
        jac = lambda v: jacobian(v, exog_vars, self.orig_endog, self.adl, exog_polys)
        fit_model = least_squares(fun, init_params, jac, method=method, ftol=ftol, xtol=xtol, gtol=gtol,
                                x_scale=x_scale, loss=loss, f_scale=f_scale, diff_step=diff_step,
                                tr_solver=tr_solver, tr_options=tr_options, jac_sparsity=jac_sparsity,
                                max_nfev=max_nfev, verbose=verbose)
        self.results = fit_model
        # generate combined dataframe of all predictior values for easier statistical evaluation
        X = pd.DataFrame(np.ones(self.orig_endog.shape[0]), columns=['Constant']).set_index(self.orig_endog.index)
        # parse the parameter values from the least squares optimization
        self.intercept = fit_model.x[0]
        params = [{'Variables': 'Constant', 'Coefficients': fit_model.x[0]}]
        thetas_iter = [thetas for key, thetas in nested_dict_iter(self.exog) if key == 'num_thetas']
        thetas_iter = list(np.cumsum(thetas_iter) - np.array(thetas_iter))
        for df in self.exog.keys():
            # parse beta parameters for weighted exogenous variables
            for var, i in zip(self.exog[df], range(1, (num_exog + 1))):
                self.exog[df][var].update({'beta': fit_model.x[i]})
                params.append({'Variables': var, 'Coefficients': fit_model.x[i]})
            # parse theta parameters for exogenous polynomial weighting
            for var, i in zip(self.exog[df], thetas_iter):
                self.exog[df][var].update({'theta(s)': fit_model.x[i + num_exog + 1:i + num_exog  + 1 + self.exog[df][var]['num_thetas']]})
            # apply polynomial weights to compress high frequency variable onto lower target frequency
            for var in self.exog[df]:
                xw = polynomial_weights(self.exog[df][var]['poly']).x_weighted(self.exog[df][var]['projection_matrix'],
                                                                               self.exog[df][var]['theta(s)'])[0]
                xw = pd.DataFrame(xw, columns=[var]).set_index(self.orig_endog.index)
                self.exog[df][var].update({'weighted': xw})
                X = pd.concat([X, xw], axis=1)
        params = pd.DataFrame(params)
        if self.adl is not None: # check if autoregressive distributed lags are included in model
                self.lambdas = fit_model.x[-self.adl.shape[1]:] # parse lambda parameters if adl exists
                params = pd.concat([params,pd.DataFrame(self.adl.columns, fit_model.x[-self.adl.shape[1]:], columns=['Variables']).rename_axis('Coefficients').reset_index()])
                X = pd.concat([X, self.adl], axis=1)
        self.params = params.set_index('Variables')
        self.X = X[params.Variables]
        return MIDASResultsWrapper(self.__dict__)


class MIDASResultsWrapper(object):
    """
    MIDAS regression model results wrapper class. Includes in -sample
    and out-of-sample prediction and statistical summary methods.
    
    Author: Yoseph Zuskin
    License: MIT 
    
    Parameters
    ----------
    model:
        Model parameters, properties, and optimization results inputted
        as a dictionary of MIDASRegressor attributes
    
    Attributes
    ----------
    orig_endog : pandas.DataFrame
        Original endogenous dependent values with datetimes index.
    endog_name : string
        Name of endogenous dependent variable.
    target_frequency: stirng
        Inferred time frequency of endogenous dependent variable.
    adl : pandas.DataFrame
        Autoregressive distributed lag terms based on ylag parameter.
    horizon: tuple
        Horizon parameters for each exogenous variable.
    exog: dictionary
        Data series and related properties of exogenous variables.
        Can be parsed using midaspy.iolib.nested_dict_iter to access
        the data and properties of all variables. For a specified
        variable, must specify its position in the exogenous input
        using the following example scheme: exog['df1']['var_name']
        for var_name in the first (or only) dataframe or series given
        in the exogenous input parameter. Will include coeficients
        and weighting parameters after fitting the model.
    num_exog : integer
        Number of exogenous independent variables used in the model.
    intercept : float
        Intercept value for the model based on optimization result.
    lambdas : numpy.array
        Coefficient values for autoregressive distributed lag terms.
    params : pandas.DataFrame
        Coefficient values from optimization results, excluding the
        polynomial weighting parameters (which can be found in exog).
    X: pd.DataFrame
        Constant and values of all weighted exogenous predictors
        and autoregressive distributed lag terms.
    """
    def __init__(self, kwarg):
        for key in kwarg:
            setattr(self, key, kwarg[key])
    
    def predict(self, exog=None, endog=None):
        """
        Generate in-sample or out-of-sample predictions.
        
        Parameters
        ----------
        exog : pandas.DataFrame, pandas.Series, list, or tuple
            Out-of-sample exogenous data. Must be in same format as original.
        endog :
            Out-of-sample endogenous data (optional, bu must be included if
            model contains autoregressive distributed lag terms and out-of-
            sample exogenous data has also been provided)
        
        Returns
        -------
        predictions : pandas.DataFrame
            Predicted target values based on in-sample or out-of-sample data
        """
        
        if exog is None and endog is None:
            diff = pd.DataFrame(-self.results.fun).set_index(self.orig_endog.index)
            prediction = pd.concat([self.orig_endog, diff], axis=1).sum(axis=1)
            return prediction.rename(self.endog_name + ' Predictions').to_frame()
        
        if type(endog) == pd.Series and endog.name == self.endog_name or type(endog) == pd.DataFrame and endog.shape[1] == 1 and endog.columns[0] == self.endog_name:
            if type(endog.index) == pd.core.indexes.datetimes.DatetimeIndex:
                new_endog = pd.DataFrame(endog)
                new_target_dates = new_endog.index
            else:
                try:
                    new_target_dates = pd.to_datetime(endog.index)
                    new_endog = pd.DataFrame(endog).set_index(new_target_dates)
                except:
                    raise ValueError('new endogenous data must have datetimes index')
            if self.adl is not None:
                adl = pd.DataFrame()
                for lag in range(1, self.adl.shape[1] + 1):
                    adl = pd.concat([adl, new_endog.shift(lag).rename(columns={new_endog.columns[0]:new_endog.columns[0]+' t-{}'.format(lag)})], axis=1)
                adl = adl.iloc[self.adl.shape[1]:, :]
                new_target_dates = new_target_dates[self.adl.shape[1]:]
                new_endog = new_endog.loc[new_target_dates, :]
        else:
            raise ValueError('new endogenous data must be series or one column data frame with same name as original')
        exog_names = [item for sublist in [list(data.keys()) for _, data in list(self.exog.items())] for item in sublist]   
        if type(exog) == pd.DataFrame and exog.shape[1] == self.num_exog or type(exog) == pd.Series and self.num_exog == 1:
            exog = pd.DataFrame(exog)
            if not set(exog.columns).issubset(exog_names):
                raise ValueError('new exogenous data column names do not match original input')
            if type(exog.index) != pd.core.indexes.datetimes.DatetimeIndex:
                raise ValueError('new exogenous data must have valid datetimes index')
            orig_exog, new_exog = [d for d in nested_dict_iter(self.exog)], dict()
            for var, i in zip(exog_names, range(0, len(orig_exog), 9)):
                projection_matrix = low_freq_projection(exog[var], orig_exog[i + 4][1],
                                                        self.horizon, new_target_dates)
                new_exog.update({var: {
                    'projection_matrix': projection_matrix,
                    'weighted': pd.DataFrame(
                        polynomial_weights(orig_exog[i + 2][1]).x_weighted(projection_matrix,orig_exog[i + 7][1])[0],
                        columns = [var]).set_index(projection_matrix.index)
                }})
                # check to ensure there is enough higher frequency data to enable
                # projection of the high-frequency variable onto the low-frequency
                var_freq = orig_exog[i + 1][1]
                if var_freq == 'A':
                    time_delta = relativedelta(years = self.horizon)
                elif var_freq == 'Q':
                    time_delta = relativedelta(months = 3 * self.horizon)
                elif var_freq == 'M':
                    time_delta = relativedelta(months = self.horizon)
                elif var_freq == 'W':
                    time_delta = relativedelta(weeks = self.horizon)
                elif var_freq == 'D':
                    time_delta = relativedelta(days = self.horizon)
                elif var_freq == 'H':
                    time_delta = relativedelta(hours = self.horizon)
                elif var_freq == 'T':
                    time_delta = relativedelta(minutes = self.horizon)
                elif var_freq == 'S':
                    time_delta = relativedelta(seconds = self.horizon)
                else: ### this should never be raised ###
                    raise SystemError('unknown frequency detection error, please log and report this to module author via GitHub')
                if new_target_dates[-1] - time_delta not in exog[var].index:
                    raise ValueError('not enough observations at the end of {} to generate last row of the lagged projection matrix'.format(var))
                available_for_row1 = exog[var][:new_target_dates[0]].shape[0]
                needed_for_all_rows = (orig_exog[i + 4][1] + self.horizon)
                if available_for_row1 >= needed_for_all_rows:
                    continue
                else:
                    raise ValueError('not enough observations available for {} to create lagged projection matrix, need to have at least {} more observations'.format(
                    var, needed_for_all_rows - available_for_row1))
        elif type(exog) == list or type(exog) == tuple:
            orig_exog, new_exog = [d for d in nested_dict_iter(self.exog)], dict()
            for var, i in zip(exog_names, range(0, len(orig_exog), 9)):
                new_exog.update({orig_exog[i][1].name: {
                    'poly': orig_exog[i + 2][1],
                    'xlag': orig_exog[i + 4][1],
                    'theta(s)': orig_exog[i + 7][1]
                }})
            for df, i in zip(exog, range(len(exog))):
                if type(df.index) != pd.core.indexes.datetimes.DatetimeIndex:
                    try:
                        exog_df = pd.DataFrame(df).set_index(pd.to_datetime(df.index))
                    except:
                        raise ValueError('element {} of new exogenous list/tutple does not have valid datetimes index'.format(i))
                exog_df = pd.DataFrame(df)
                for var in exog_df.columns:
                    if var not in exog_names:
                        continue # maybe raise a warning here?
                    projection_matrix = low_freq_projection(exog_df[var], new_exog[var]['xlag'],
                                                            self.horizon, new_target_dates)
                    xw = pd.DataFrame(polynomial_weights(new_exog[var]['poly']).x_weighted(
                        projection_matrix, new_exog[var]['theta(s)'])[0], columns=[var]).set_index(new_target_dates)
                    new_exog[var].update({'projection_matrix': projection_matrix, 'weighted': xw})
        values = pd.concat([df for _, df in nested_dict_iter(new_exog) if _ == 'weighted'], axis=1)
        values = pd.concat([pd.DataFrame(np.ones(values.shape[0]), columns=['Constant']).set_index(values.index),
                            values], axis=1)
        if self.adl is not None:
            values = pd.concat([values, adl], axis=1)
        values = values[self.params.index.tolist()]
        prediction = np.dot(values.values, self.params.values).reshape((len(new_target_dates),))
        return pd.Series(prediction).rename(self.endog_name + ' Predictions').to_frame()
                
    def conf_int(cls, alpha=.05, sig_dig=3):
        """
        Return confidence interval of beta coefficients given alpha. Uses students' t-test.
        Based on the code for the statsmodels..regression.linear_model.OLS.conf_int method.
        
        Parameters
        ----------
        alpha : float, optional
            The significance level for the confidence interval. The default `alpha` = .05
            returns a 95% confidence interval.
        sig_dig : int, optional
            Number of significant digitals to show in the output table. The default is 3.
            
        Returns
        -------
        conf_int : pandas.DataFrame
            Each row contains [lower, upper] limits of the confidence interval for the
            corresponding parameter. The first column contains all lower, the second column
            contains all upper limits.
        """
        mse = (sum((cls.orig_endog.values - cls.predict().values) ** 2)) / (cls.X.shape[0] - cls.X.shape[1])
        var_b = mse * (np.linalg.inv(np.dot(cls.X.T.values, cls.X.values)).diagonal())
        sd_b = np.sqrt(var_b)
        df_resid = cls.orig_endog.shape[0] - 1
        q = t.ppf(1 - alpha / 2, df_resid)
        params = cls.params.iloc[:, 0].values
        lower = params - q * sd_b
        upper = params + q * sd_b
        conf_int = pd.DataFrame(np.column_stack([lower, upper]),
                                columns=[alpha / 2, 1 - alpha / 2]).set_index(
            cls.params.index.values).round(sig_dig)
        return conf_int
    
    def score(cls):
        """
        Return coefficient of determination R^2 of the prediction.
        """
        return np.corrcoef(cls.orig_endog.iloc[:, 0].values, cls.predict().iloc[:, 0].values)[0, 1] ** 2
    
    def adj_r_squared(cls):
        """
        Return coefficient of determination, adjusted for numbers of observations
        and model parameters (both regressor coefficients and weighting parameters).
        """
        n = cls.orig_endog.shape[0]
        p = len(cls.results.x) - 1 # excludes constant coefficient
        return 1 - (1 - cls.score()) * (n - 1) / (n - p - 1)
    
    def rse(cls):
        """
        Return residual standard error score of the prediction.
        """
        rss = sum((cls.orig_endog.values - cls.predict().values) ** 2)[0]
        n = cls.orig_endog.shape[0]
        p = len(cls.results.x) - 1 # excludes constant coefficient
        return np.sqrt(rss / (n - p - 1))
    
    def f_stat(cls):
        """
        Return F-statistic of the prediction.
        """
        tss = sum((cls.orig_endog.values - cls.orig_endog.mean()[0]) ** 2)[0]
        rss = sum((cls.orig_endog.values - cls.predict().values) ** 2)[0]
        n = cls.orig_endog.shape[0]
        p = len(cls.results.x) - 1 # excludes constant coefficient
        return ((tss - rss) / p) / (rss / (n - p - 1))
    
    def significance(cls, alpha=.05, sig_dig=3):
        """
        Return summary of variable significance. Similar to middle part of
        statsmodels.api.OLS.fit.summary method.
        
        Source: https://stackoverflow.com/questions/27928275/
        
        Parameters
        ----------
        alpha : float, optional
            The significance level for the confidence interval. The default `alpha` = .05
            returns a 95% confidence interval.
        sig_dig : int, optional
            Number of significant digitals to show in the output table. The default is 3.
            
        Returns
        -------
        significance : pandas.DataFrame
             Table showing standard  error, t-test, p-values, and confidence interval for
             the beta coefficient of each variable (using student's t-test; 95% default).
        """
        mse = (sum((cls.orig_endog.values - cls.predict().values) ** 2)) / (cls.X.shape[0] - cls.X.shape[1])
        var_b = mse * (np.linalg.inv(np.dot(cls.X.T.values, cls.X.values)).diagonal())
        sd_b = np.sqrt(var_b)
        ts_b = cls.params.iloc[:, 0].values / sd_b
        p_values = [2 * (1 - t.cdf(np.abs(i),(len(cls.X.values) - 1))) for i in ts_b]
        df_resid = cls.orig_endog.shape[0] - 1
        q = t.ppf(1 - alpha / 2, df_resid)
        params = cls.params.iloc[:, 0].values
        lower = params - q * sd_b
        upper = params + q * sd_b
        significance = pd.DataFrame([cls.params.Coefficients.values,sd_b, ts_b, p_values,
                                    lower, upper]).T.set_index(cls.params.index.values).round(sig_dig)
        significance.columns = ['coef', 'std err', 't', 'P>|t|', str(alpha / 2), str(1 - alpha / 2)]
        return significance