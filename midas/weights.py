import numpy as np
from scipy.special import gamma
from abc import ABC, abstractmethod

def weights_init(poly):
    r"""Initiate beta polynomial or exponential Almon polynomial exogenous weighting.
    
    Args:
        poly (str): ``exp_almon``, ``beta``, or ``hyperbolic`` polynomial weight method.
        
    Returns:
        poly_class (midas.weights.WeightMethod): Instantiated polynomial weighting class.
    """
    poly_class = {'beta': (1., 5.), 'exp_almon': (-.5, .01), 'hyperbolic': (.1)}
    return poly_class[poly]

class WeightMethod(ABC):
    r"""Polynominal weighting method abbstract base class.
    """
    @abstractmethod
    def __init__(self):
        pass
    @abstractmethod
    def weights(self):
        pass
    def x_weighted(self, x, thetas):
        r"""Apply weights on lagged projection matrix.
        
        Parameters
        ----------
        x : numpy.ndarray
            Lagged projection matrix.
        thetas : list
            Polymonial weighting parameters.
        
        Returns
        -------
        xw : numpy.ndarray
            Lagged projection matrix multiplied with the weights.
        ws : numpy.ndarray
            Matrix of the weights that were applied on each row.
        """
        self.theta1, self.theta2 = thetas
        w = self.weights(x.shape[1])
        xw = np.dot(x, w)
        ws = np.tile(w.T, (x.shape[1], 1))
        return xw, ws

class BetaWeights(WeightMethod):
    def __init__(self, theta1, theta2):
        super(BetaWeights, self).__init__()
        self.theta1 = theta1
        self.theta2 = theta2
    def weights(self, nlags):
        r"""Beta polynomial weighting method.
        
        Parameters
        ----------
        nlags : int
            Number of laged values in rows of the projection matrix.
        
        Returns
        -------
        w : numpy.array
            Instantianted nlag weights vector to be applied on each row of the
            lagged projection matrix. Used during the x_weighted method.
        """
        eps = np.spacing(1)
        u = np.linspace(eps, 1.0 - eps, nlags)
        w = u ** (self.theta1 - 1) * (1 - u) ** (self.theta2 - 1)
        return w / sum(w)
    @property
    def num_params(self):
        return 2
    
class ExpAlmonWeights(WeightMethod):
    def __init__(self, theta1, theta2):
        super(ExpAlmonWeights, self).__init__()
        self.theta1 = theta1
        self.theta2 = theta2

    def weights(self, nlags):
        r"""Exponential Almon polynomials weighting method.

        Parameters
        ----------
        nlags : int
            Number of laged values in rows of the projection matrix.
        
        Returns
        -------
        w : numpy.array
            Instantianted nlag weights vector to be applied on each row of the
            lagged projection matrix. Used during the x_weighted method.
        """
        u = np.arange(1, nlags + 1)
        w = np.exp(self.theta1 * u + self.theta2 * u ** 2)
        return w / sum(w)
    @property
    def num_params(self):
        return 2

class HyperbolicWeights(WeightMethod):
    def __init__(self, theta):
        super(HyperbolicWeights, self).__init__()
        self.theta = theta
    def weights(self, nlags):
        r"""Hyperbolic scheme weighting method.

        Parameters
        ----------
        nlags : int
            Number of laged values in rows of the projection matrix.
        
        Returns
        -------
        w : numpy.array
            Instantianted nlag weights vector to be applied on each row of the
            lagged projection matrix. Used during the x_weighted method.
        """
        u = np.arange(1, nlags + 1)
        g = gamma(u + self.theta) / (gamma(u + 1) * gamma(self.theta))
        return g / sum(g)
    @property
    def num_params(self):
        return 1