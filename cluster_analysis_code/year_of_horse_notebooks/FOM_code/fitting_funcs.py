#############################################################
#
#          All fitting functions for FOM
#                if you want to add new ones, do it here :)
#
# jwaiton 170426
#
#############################################################

import numpy as np
import iminuit
from   iminuit import Minuit
import probfit
from scipy.stats import crystalball, norm
import traceback
import matplotlib.pyplot as plt

# misc functions

def add_element(dict, key, value):
    if key not in dict:
        dict[key] = value


def ratio_error(f, a, b, a_error, b_error):
    '''
    docs for this online, need to move them over
    '''

    return f*np.sqrt((a_error/a)**2 + (b_error/b)**2)


def fom_error(a, b, a_error, b_error):
    '''
    docs for this online, move them over
    derived in joplin notes 11/04/24
    '''
    
    element_1 = np.square(a_error/np.sqrt(b))
    element_2 = np.square((b_error * a) /(2*(b**(3/2))))
    return np.sqrt(element_1 + element_2)

# pdf functions


def crystal_ball_signal(x, beta, m, loc, scale, ns = None):
    '''
    signal function now defined by a crystal ball
    Args:
        x       : numpy.ndarray
                  Independent variable (e.g., energy values).
        ns      : float
                  Normalization factor for the signal component.
                  Can be disabled to ignore it
        B       : float
                  Background parameter  related to the exponential decay.
        M       : float
                  Mean value of the crystal ball distribution, representing the peak position.
    Returns:
        numpy.ndarray
        The evaluated signal function values for the given input parameters.
    '''
    
    if ns is None:
        return (crystalball.pdf(x, beta, m, loc = loc, scale = scale))
    else:
        return ns * (crystalball.pdf(x, beta, m, loc = loc, scale = scale))


def sig_bck_func(x, B, M, tau, loc, scale, ns = None, nb = None):
    '''
    signal and background (exponential) functions combined
    '''
    return (sig_func(x, ns, B, M, loc, scale) + FOM_func.bck_func(x, nb, tau))


def gaussian_no_N(x, loc, scale):
    '''
    gaussian non normalised
    
    loc   --> mu
    scale --> sigma
    '''
    
    return norm.pdf(x, loc = loc, scale = scale)

def exp_no_N(x, tau):
    '''
    expeonential non normalised
    '''
    
    return np.exp(-x/ tau)
    

def linear(x, m, c):
    '''
    linear projection for the data
    '''
    return m*x + c
    
    
##################################################################
#####                    MINUIT FITTING FUNCS                #####
##################################################################


def gaussian_fit(data, fitting_info, plot = False):
    '''
    fit a gaussian to the data with unbinned likelihood fit
    '''
    # take data
    fit_data = data['energy'].to_numpy()
    
    # make gaussian fit 
    gauss_norm     = probfit.Normalized(gaussian_no_N, fitting_info['fit_range'])
    gauss_norm_ext = probfit.Extended(gauss_norm, extname = 'Ng')

    lh_g   = probfit.UnbinnedLH(gauss_norm_ext, data['energy'].to_numpy(), extended = True)
    vals_g = [len(fit_data), 1.59, 0.004]        # defaults
    nm_g   = ['Ng', 'loc', 'scale']               # labels


    m_g    = Minuit(lh_g, **dict(zip(nm_g, vals_g)),
                    limit_loc    = (fitting_info['fit_range'][0], fitting_info['fit_range'][1]),
                    limit_Ng     = (0, None),
                    limit_scale  = (0, 1))
    m_g.print_level = 1                         # show progress
    m_g.migrad()

    if plot:
        try:
            plt.clf()
        except Exception as e:
            print('No plot to clear')

        plt.xlabel('Track energy (Mev)')
        plt.ylabel('Counts/bin')
        m_g.show(bins=fitting_info['bins']+1, parts = True)

    # pull out relevant info
    fit_params = {}
    [add_element(fit_params, m_g.params[i][1], m_g.params[i][2]) for i in range(len(m_g.params))]

    return (fit_params['loc'], fit_params['scale'])


def sb_fit(data, sig_pdf, bck_pdf, fitting_info, seeds, plot = False):
    '''
    apply the combined signal and background fits to the data
    '''

    blob_data = data['energy'].to_numpy()
    
    # combine s&b
    pdf_sb = probfit.AddPdf(sig_pdf, bck_pdf)
    lh_sb  = probfit.BinnedLH(pdf_sb, blob_data, extended = True)

    # seed extraction 
    vals = list(seeds.values())
    keys = list(seeds.keys())
    
    # update seed with half of the current data vals
    try:
        seeds.update({'Nb': len(blob_data)/2, 'Ns': len(blob_data)/2}) 
    except Exception as e:
        print('Couldnt update seeds')
        print(traceback.format_exc())    

    print('Initial parameters')
    for i in range(len(vals)):
        print(f"{keys[i]}: {vals[i]:.4f} \u00B1 {list(np.diag(np.zeros_like(vals)))[i][i]:.4f}") 

    # minuit shenans
    m_sb = Minuit(lh_sb, **seeds,
                  fix_loc = True, # this is assuming your fit is gaussian using loc and scale as mu and sigma
                  #limit_tau = (0.1, None),
                  limit_Nb    = (0, None),
                  limit_Ns    = (0, None),
                  limit_scale = (1e-4, None),
                  print_level = 2) 
    m_sb.migrad()

    
    if plot: # hardcoded for linear and gaussian.
        heights, bins, _ = plt.hist(blob_data, fitting_info['bins'])
        bin_width = bins[1] - bins[0]
        x = np.linspace(fitting_info['fit_range'][0], fitting_info['fit_range'][1], 100)
    
        v = dict(zip(m_sb.parameters, m_sb.args))
        Ns = v['Ns']
        Nb = v['Nb']
    
        sig = Ns * sig_pdf(x, v['loc'], v['scale']) * bin_width
        bck = Nb * bck_pdf(x, v['m'], v['c']) * bin_width
    
        plt.plot(x, sig + bck, label='total')
        plt.plot(x, sig, label='signal')
        plt.plot(x, bck, label='background')
        plt.xlabel("Track energy (MeV)")
        plt.ylabel("Counts/bin")
        plt.legend()
        plt.show()
    
    # pull out relevant info
    fit_params = {}
    [add_element(fit_params, m_sb.params[i][1], m_sb.params[i][2]) for i in range(len(m_sb.params))]
    
    print('Fit parameters')
    print(fit_params)
    
    return (fit_params['Ns'], fit_params['Nb'])

