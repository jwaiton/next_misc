'''

FOM fitting functions

jwaiton 240426

'''
import numpy  as np
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


def gaussian(x, mu, sigma, A):
    '''
    Simple gaussian function
    '''
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def gaussian_fit(data, bins):
    '''

    '''
    counts, bin_edges = np.histogram(data, bins=bins)
    bin_centres       = (bin_edges[:-1] + bin_edges[1:]) / 2

    p0             = [np.mean(data), np.std(data), np.max(counts)]
    popt, pcov     = curve_fit(gaussian, bin_centres, counts, p0=p0)
    mu, sigma, A   = popt

    return mu, sigma, A
