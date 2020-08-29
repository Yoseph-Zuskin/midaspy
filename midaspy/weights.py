from abc import ABC, abstractmethod
import numpy as np

def polynomial_weights(poly, weights):
    r"""Initiate beta polynomial or exponential Almon polynomial exogenous weighting.
    
    Args:
        poly (str): 'exp_almon', 'beta', or 'hyperbolic' polynomial weight method.
        thetas (tuple or list): Instantiation parameters
        
    Returns:
        poly_class (midaspy.weights.WeightMethod): Instantiated weight method.
    """
    poly_class = {
        'beta': BetaWeights(1., 5.),
        'exp_almon': ExpAlmonWeights(-1., 0.),
        'hyperbolic': HyperbolicWeights(.1)
                 }
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
    @abstractmethod
    def x_weighted(self):
        pass

class BetaWeights(WeightMethod):
    def __init__(self, theta1, theta2, theta3=None):
        super(BetaWeightsl, self).__init__()
        self.theta1 = theta1
        self.theta2 = theta2
        self.theta3 = theta3
    def weights(self, nlags):
        r"""Beta polynomial weighting method.
        
        Args:
            nlags (int): Number of laged values in rows of the projection matrix.
        
        Returns:
            w (numpy.array): Instantianted nlag weights vector to be applied on
                each row of the projection matrix.
        """
        eps = np.spacing(1)
        u = np.linspace(eps, 1.0 - eps, nlags)
        w = u ** (self.theta1 - 1) * (1 - u) ** (self.theta2 - 1)
        w = w / sum(w)
        if self.theta3 is not None:
            w += self.theta3
            return w / sum(w)
        return w
    def x_weighted(self, x, params):
        r"""
        """
        self.theta1, self.theta2, self.theta3 = params
        w = self.weights(x.shape[1])
        xw = np.dot(x, w)
        w = np.tile(w.T, (x.shape[1], 1))
        return xw, w
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
        Hyperbolic (gamma) polynomial weighting method.

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