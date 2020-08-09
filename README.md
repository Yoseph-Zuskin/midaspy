# Mixed Data Sampling (MIDAS) Modeling in Python
Libirary usage tutorial: <a href="https://colab.research.google.com/github/Yoseph-Zuskin/midaspy/blob/master/example/monthly_CPI_daily_stock_index.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## Current Features
- Beta, Exponential Almon, and Hyperbolic scheme polynomial weighting methods.
- Lagged matrix generator function to project higher frequency data onto lower frequency.
- Flexible MIDAS ordinary least squares regressor model and results wrapper classes.
- Basic statisical summary methods like R^2 score and variable significance (t-test p-values).

## Future Work
- Improve efficiency of exogenous lagged projection generator function.
- Enable horizon to be set for each exogenous variable separately.
- Enable prelagged variables to be used for faster model fitting.
- More comprehensive statistical summary method simialr to statsmodels.api.OSL().fit().summary().
- Create a Flexible MIDAS logistic classifier model class.
