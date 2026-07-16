'''

FOM fitting functions

jwaiton 240426

'''
import numpy  as np
import numpy.polynomial.chebyshev as cheb
import pandas as pd

from scipy.optimize import curve_fit


import zfit
import tensorflow as tf
from zfit.interface import ZfitPDF


def gaussian_no_N(obs,
                  mu_config    = None,
                  sigma_config = None,
                  name_suffix  = ""):
    '''
    Gaussian model from zfit
    https://zfit.readthedocs.io/en/stable/user_api/_generated/param/zfit.param.Parameter.html
    https://zfit.readthedocs.io/en/stable/user_api/pdf/_generated/basic/zfit.pdf.Gauss.html#zfit.pdf.Gauss

    obs (zfit.Space)      : Space the gaussian will cover

    mu_config (dict)      :  Containing:
                               value     - default value
                               lower     - lower limit of value
                               upper     - upper limit of value
                               floating  - Fixed or not (bool)
                               label     - /
                               stepsize  - Stepsize for minimisation

    sigma_config (dict)   :  Same as above

    name_suffix (str)     :  String to avoid overlapping names of zfit parameters


    '''
    # force names
    mu_name    = f'mu{name_suffix}'
    sigma_name = f'sigma{name_suffix}'

    # set default to 1
    if mu_config is None:
        mu = zfit.Parameter(mu_name, 1)
    else:
        mu = zfit.Parameter(mu_name, **mu_config)

    if sigma_config is None:
        sigma = zfit.Parameter(sigma_name, 1)
    else:
        sigma = zfit.Parameter(sigma_name, **sigma_config)


    return zfit.pdf.Gauss(obs = obs, mu=mu, sigma=sigma)


def exp_no_N(obs,
             lambda_config = None,
             name_suffix   = ""):
    '''
    Exponential model from zfit
    https://zfit.readthedocs.io/en/stable/user_api/_generated/param/zfit.param.Parameter.html
    https://zfit.readthedocs.io/en/stable/user_api/pdf/_generated/basic/zfit.pdf.Exponential.html#zfit.pdf.Exponential

    obs (zfit.Space)          : Space the gaussian will cover

    lambda_config (dict)      :  Containing:
                               value     - default value
                               lower     - lower limit of value
                               upper     - upper limit of value
                               floating  - Fixed or not (bool)
                               label     - /
                               stepsize  - Stepsize for minimisation

    name_suffix (str)     :  String to avoid overlapping names of zfit parameters
    '''
    exp_name = f'lambda{name_suffix}'

    if lambda_config is None:
        tau = zfit.Parameter(exp_name, 1)
    else:
        tau = zfit.Parameter(exp_name, **lambda_config)

    return zfit.pdf.Exponential(obs = obs, lam=tau)


def poly_no_N(obs,
              a_config = None,
              name_suffix = ""):
    '''
    Polynomial model from zfit
    https://zfit.readthedocs.io/en/stable/user_api/_generated/param/zfit.param.Parameter.html
    https://zfit.readthedocs.io/en/stable/user_api/pdf/_generated/polynomials/zfit.pdf.Chebyshev.html#zfit.pdf.Chebyshev

    obs (zfit.Space)          : Space the gaussian will cover

    a,b,c_config (dict)      :  Containing:
                               value     - default value
                               lower     - lower limit of value
                               upper     - upper limit of value
                               floating  - Fixed or not (bool)
                               label     - /
                               stepsize  - Stepsize for minimisation

    name_suffix (str)     :  String to avoid overlapping names of zfit parameters
    '''

    a_name = f'polya{name_suffix}'
    b_name = f'polyb{name_suffix}'

    if a_config is None:
        a = zfit.Parameter(a_name, 0)
    else:
        a = zfit.Parameter(a_name, **a_config)


    return zfit.pdf.Chebyshev(obs = obs, coeffs = [a])

def exponential(x, tau, B):
    return B * np.exp(tau * x)


def gaussian(x, mu, sigma, A):
    '''
    Simple gaussian function
    '''
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def gaussian_exp(x, mu, sigma, A, B, tau):
    gauss = A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    exp   = B * np.exp(tau * x)

    return gauss + exp

def polynomial(x, a, N):
    return N * cheb.chebval(x, [1.0, a])

def gaussian_fit(data, bins):
    '''

    '''
    counts, bin_edges = np.histogram(data, bins=bins)
    bin_centres       = (bin_edges[:-1] + bin_edges[1:]) / 2

    p0             = [np.mean(data), 0.005, np.max(counts)/2, np.max(counts)/2, 0.001]
    popt, pcov     = curve_fit(gaussian_exp, bin_centres, counts, p0=p0)
    mu, sigma, A, _, _   = popt

    return mu, sigma, A


def exp_fit(data, bins):
    '''

    '''
    counts, bin_edges = np.histogram(data, bins=bins)
    bin_centres       = (bin_edges[:-1] + bin_edges[1:]) / 2

    p0             = [0.00001, np.sum(counts)]
    popt, pcov     = curve_fit(exponential, bin_centres, counts, p0=p0, bounds = [[0, 0],[np.inf, np.inf]])
    tau, B         = popt

    return tau, B


def polynomial_fit(data, bins, fit_range=(1.4, 1.8)):
    counts, bin_edges = np.histogram(data, bins=bins)
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Scale to [-1, 1] using the global fit limits, not data-dependent limits
    x_min, x_max = fit_range
    bin_centres_scaled = 2.0 * (bin_centres - x_min) / (x_max - x_min) - 1.0

    p0 = [0.0, counts.max()] # a=0 (linear), b=0 (quadratic), N=amplitude
    lower_bounds = [-1.0, 0]
    upper_bounds = [1.0, np.inf]

    popt, pcov = curve_fit(
        polynomial,
        bin_centres_scaled,
        counts,
        p0=p0,
        bounds=(lower_bounds, upper_bounds)
    )
    a, N = popt
    return a, N
