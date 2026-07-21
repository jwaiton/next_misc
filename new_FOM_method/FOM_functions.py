'''

FOM function template

jwaiton 160526
'''

import numpy  as np
import pandas as pd

import matplotlib.pyplot as plt
import zfit
import traceback
import csv

from pathlib import Path

import fitting_functions   as fitf
import cutting_functions   as cutf
import plotting_functions  as plotf
import error_functions     as errf

def build_model(signal_func, background_func, obs, seeds, name_suffix=""):
    '''
    generate signal and background model

    signal_func (func)     :  function as described in fitting_functions.py
    background_func (func) :  function as described in fitting_functions.py
    obs (zfit.Space)       :  zfit space over which fit is done
    seeds (dict)           :  seeds for signal and background, better described
                              in fitting_functions.py
    name_suffix (str)      :  suffix to avoid zfit parameter overwriting

    '''
    sig_pdf = signal_func         (obs, **seeds.get("signal", {}), name_suffix=name_suffix)
    bck_pdf = background_func     (obs, **seeds.get("background", {}), name_suffix=name_suffix)

    # extend
    Ns = zfit.Parameter(f"Ns{name_suffix}",
                        value = seeds.get("Ns", 5000),
                        lower = 0,
                        upper = None)

    Nb = zfit.Parameter(f"Nb{name_suffix}",
                        value = seeds.get("Nb", 5000),
                        lower = 0,
                        upper = None)

    sig_ext = sig_pdf.create_extended(Ns)
    bck_ext = bck_pdf.create_extended(Nb)

    model = zfit.pdf.SumPDF([sig_ext, bck_ext])

    return model, Ns, Nb, sig_pdf, bck_pdf


def FOM(data, signal_func, background_func,
        cut_list     = None,
        seeds        = None,
        fitting_info = None,
        plot         = False,
        output_path  = None,
        label        = None,
        verbose      = True):


    '''
    FOM function, applying the FOM fit as defined
    Calculated the FOM per blob 2 cut, then saves the data to a file


    Args:
        data             :  pd.DataFrame
                            Dataframe containing the relevant Tracking/Tracks data

        signal_func      :  function
                            Signal function passed through
        background_func  :  function
                            Background function passed through
        cut_list         :  list
                            List of all the blob 2 values to cut
        seeds            :  dict
                            Dictionary of all seeds values for signal and background,
                            as well as number of signal and number of background
        fitting_info     :  dict
                            Dictionary containing number of bins,
                            the fit_range (energy-wise) which is a tuple,
        plot             :  bool
                            Whether or not to plot the results (very spammy)
        output_path      :  str
                            Output path of the FOM data
        label            :  str
                            Label for the plot output
    '''

    # handle the none cases with defaults

    # cut_list
    if cut_list is None:
        cut_list = np.linspace(0, 0.5, 31)

    # fitting_info
    if fitting_info is None:
        print('Fitting info not provided, setting defaults')
        fitting_info = dict(
                            fit_range = (1.4, 1.8),
                            bins      = 80)

    if label is None:
        label = 'FOM'

    # generate pdfs and space
    obs  = zfit.Space('energy_range', limits = fitting_info['fit_range'])


    gaussian_fit_range = (1.55, 1.65)
    # gaussian fit prior
    gauss_hdst = cutf.energy_cuts(data, *gaussian_fit_range)
    mu_seed, sigma_seed, A = fitf.gaussian_fit(gauss_hdst['energy'].to_numpy(), bins = 15)
    if verbose:
        print(f'Initial gaussian and sigma: {mu_seed}, {sigma_seed}')

    if plot:
        plotf.plot_hist(gauss_hdst, binning = 20, output = False, log = False, title = 'Gaussian pre-fit')
        # gauss fit
        x_space = np.linspace(*gaussian_fit_range, 200)
        y_fit   = fitf.gaussian(x_space, mu_seed, sigma_seed, A)
        plt.plot(x_space, y_fit, 'r-', label = f'Fit:\nmu: {mu_seed}\nsigma={sigma_seed}')
        plt.show()

    # exponential/polynomial fit prior
    exp_hdst_lower = cutf.energy_cuts(data, 0, gaussian_fit_range[0])
    exp_hdst_upper = cutf.energy_cuts(data, gaussian_fit_range[1], data['energy'].max())
    exp_hdst       = pd.concat([exp_hdst_lower, exp_hdst_upper], ignore_index = True, sort = False)

    ####a_seed, c_seed = fitf.polynomial_fit(exp_hdst['energy'].to_numpy(), bins = 100)
    tau_seed, B    = fitf.exp_fit(exp_hdst['energy'].to_numpy(), bins = 100)
    if verbose:
       print(f'Initial exponential seed: {tau_seed}, B: {B}')
       ####print(f'Initial polynomial seed: a = {a_seed}, N = {c_seed}')

    if plot:
        plotf.plot_hist(data, binning = 100, output = False, log = False, title = 'exp pre-fit')
        # gauss fit
        x_space = np.linspace(1.4, 1.8, 500)
        y_fit   = fitf.exponential(x_space, tau_seed, B)
        plt.plot(x_space, y_fit, 'r-', label = f'Fit:\ntau: {tau_seed}\nB={B}')
        ####y_fit   = fitf.polynomial(x_space, a_seed, c_seed)
        ####plt.plot(x_space, y_fit, 'r-', label = f'Fit:\na: {a_seed}\nN: {c_seed}')
        plt.show()

    # update the seeds
    if seeds is not None:
        seeds['signal']['mu_config'].update({"value": mu_seed, "floating" : False})
        seeds['signal']['sigma_config'].update({"value": sigma_seed, "floating" : False})
        seeds['background']['lambda_config'].update({"value": tau_seed, "floating" : False})
        ####seeds['background']['a_config'].update({"value": a_seed, "floating" : False})


    # initialise minimiser
    minimiser = zfit.minimize.Minuit(verbosity = 0)

    # setup storage lists
    e, b, ns_l, nb_l, fom, fom_err, e_err, b_err = [[] for _ in range(8)]

    for i, cut in enumerate(cut_list):
        try:
            blob_data = data[(data['eblob2'] > cut)]
            zfit_data = zfit.Data.from_numpy(array=blob_data['energy'].to_numpy(), obs = obs)
            if verbose:
                print('=' * 30)
                print(f'Blob cut: {cut} MeV')
                print('=' * 30)

            # generate model
            model, Ns, Nb, sig_pdf, bck_pdf = build_model(signal_func,
                                              background_func,
                                              obs,
                                              seeds,
                                              name_suffix = f"_{i}")

            # fit da thing
            nll    = zfit.loss.ExtendedUnbinnedNLL(model = model, data = zfit_data)
            result = minimiser.minimize(nll)
            result.hesse()

            # extract info
            ns_total = result.params[Ns]['value']
            nb_total = result.params[Nb]['value']


            # limit integral over the peak
            integral_range = (mu_seed - sigma_seed*3, mu_seed + sigma_seed*3)
            sub_obs = zfit.Space('energy_range', limits = integral_range)

            ns = ns_total * sig_pdf.integrate(sub_obs).numpy()
            nb = nb_total * bck_pdf.integrate(sub_obs).numpy()

            if verbose:
                print(f'Printing over integral range {integral_range}')

            ns_l.append(ns)
            nb_l.append(nb)

            e_check   = ns / ns_l[0]
            b_check   = nb / nb_l[0]
            fom_check = e_check / np.sqrt(b_check)

            e.append(e_check)
            b.append(b_check)
            fom.append(fom_check)

            e_err.append(errf.ratio_error(e_check, ns, ns_l[0], np.sqrt(ns), np.sqrt(ns_l[0])))
            b_err.append(errf.ratio_error(b_check, nb, nb_l[0], np.sqrt(nb), np.sqrt(nb_l[0])))
            fom_err.append(errf.fom_error(e_check, b_check, e_err[i], b_err[i]))

            if verbose:
                print(f'Signal events: {ns}\nBackground events: {nb}')
                print(f'Total events by addition: {nb_total+ns_total}\nTotal events by row counting: {blob_data.event.nunique()}')
                print(f"Total events by addition in ROI: {nb+ns}\nTotal events by row counting in ROI: {blob_data[(blob_data['energy'] > integral_range[0]) & (blob_data['energy'] < integral_range[1])].event.nunique()}")

                print(f'FOM: {fom_check} +/- {fom_err[-1]}')
                print(f'=' * 30)
                print('Errors of each components of the fit.')
                # Print errors for floating model parameters (e.g., step_loc or frac)
                for param in result.params:
                    if param not in (Ns, Nb):
                        val = result.params[param]['value']
                        # Check if hesse calculated an error for this param
                        err_dict = result.params[param].get('hesse', {})
                        err_val = err_dict.get('error', 'N/A')
                        print(f'{param.name}: {val:.4f} +/- {err_val}')

            # reset for next iteration
            zfit.param.set_values(model.get_params(), result)

            # visualise the fit if asked for
            if plot:
                sig_ext, bck_ext = model.pdfs
                x_space = np.linspace(*fitting_info['fit_range'], 200)
                x_zfit  = zfit.Data.from_numpy(obs = obs, array = x_space)

                fig = plt.figure(figsize=(16, 12))
                gs  = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
                ax_main = fig.add_subplot(gs[0])
                ax_res  = fig.add_subplot(gs[1], sharex=ax_main)

                counts, bins, patches = ax_main.hist(blob_data['energy'].to_numpy(),
                                        bins = 200,#fitting_info['bins'],
                                        range = fitting_info['fit_range'],
                                        alpha = 0.6,
                                        label = 'Data')
                # take scale from first plot, use across all.
                if i == 0:
                    y_min, y_max = ax_main.get_ylim()
                else:
                    ax_main.set_ylim([y_min, y_max])


                bin_width = (fitting_info['fit_range'][1] - fitting_info['fit_range'][0]) / 200#fitting_info['bins']

                ax_main.plot(x_space, sig_ext.pdf(x_zfit) * ns_total * bin_width, label='Signal', linestyle='dashed')
                ax_main.plot(x_space, bck_ext.pdf(x_zfit) * nb_total * bin_width, label='Background')

                full_fit = (sig_ext.pdf(x_zfit) * ns_total + bck_ext.pdf(x_zfit) * nb_total) * bin_width

                ax_main.plot(x_space, full_fit, label='Total', linestyle='dashdot')
                ax_main.legend()
                plt.setp(ax_main.get_xticklabels(), visible=False)  # hide top x-axis labels

                # residuals
                res = (full_fit - counts) / np.sqrt(np.where(counts > 0, counts, 1))
                ax_res.bar(x_space, res, width = bin_width)
                ax_res.axhline(0, color='black', linewidth=0.8, linestyle='--')
                # gridlines
                ax_res.yaxis.set_major_locator(plt.MultipleLocator(3))
                ax_res.yaxis.set_minor_locator(plt.MultipleLocator(1))
                ax_res.grid(axis='y', which='both', linestyle='--', linewidth=0.7, alpha=0.7)

                ax_res.set_ylabel(r'$(f - n)/\sqrt{n}$', fontsize=20)
                ax_res.set_ylim(-5, 5)          # adjust to taste; ±3 is a common choice
                ax_res.set_xlabel('Reconstructed Energy (MeV)')
                # scale factor on data
                #scale  = (fitting_info['fit_range'][1] - fitting_info['fit_range'][0]) / fitting_info['bins']
                #plt.plot(x_space, sig_ext.pdf(x_zfit) * ns_total * scale, label = 'Signal', linestyle = 'dashed')
                #plt.plot(x_space, bck_ext.pdf(x_zfit) * nb_total * scale, label = 'Background')
                #plt.plot(x_space, (sig_ext.pdf(x_zfit) * ns_total + bck_ext.pdf(x_zfit) * nb_total) * scale, label = 'Total', linestyle = 'dashed')
                #plt.xlabel('Reconstructed Energy (MeV)')
                ax_main.set_ylabel('Counts')
                ax_main.set_xlim([1.4, 1.8])
                plt.savefig(f'{output_path}/FOM_fit_{cut:.2f}MeV.pdf')
                plt.savefig(f'{output_path}/{cut:.2f}MeV.png')
                plt.show()

                # chi2
                chi2      = np.sum(res**2)
                ndof      = len(counts) - len(result.params) # number of degrees of freedom
                chi2_ndof = chi2 / ndof
                if verbose:
                    print(f'Chi squared over degrees of freedom: {chi2_ndof}')


        except Exception as err:
            print("FIT BROKE!")
            print(traceback.format_exc())
            for lst in [ns_l, nb_l, e, b, fom, e_err, b_err, fom_err]:
                lst.append(np.array([-9999]))


    # flatten for output
    fom_err = np.array(fom_err).flatten()
    fom     = np.array(fom).flatten()
    ns_l    = np.array(ns_l).flatten()
    nb_l    = np.array(nb_l).flatten()
    e       = np.array(e).flatten()
    b       = np.array(b).flatten()
    b_err   = np.array(b_err).flatten()
    e_err   = np.array(e_err).flatten()
    if output_path is not None:
        # ensure output_path exists, if not, make it
        pth = Path(output_path)
        pth.mkdir(parents=True, exist_ok=True)
        # plot
        try:
            plt.errorbar(cut_list, fom, yerr=fom_err, linestyle="dashed")
            plt.legend()
            plt.title(f"FOM {label}")
            plt.xlabel("Blob 2 energy threshold (MeV)")
            plt.ylabel("FOM")
            plt.savefig(f"{'/'.join(output_path.split('/')[:-1])}/FOM_fit.png")
            plt.savefig(f"{'/'.join(output_path.split('/')[:-1])}/FOM_fit.pdf")
            plt.close()
        except Exception as err:
            traceback.print_exc()
        try:
            # write csv
            with open(f"{output_path}/FOM.csv", "w") as f:
                writer = csv.writer(f)
                writer.writerows(zip(cut_list,          [float(x) for x in fom]))
                writer.writerows(zip(cut_list,          [float(x) for x in fom_err]))
                writer.writerows(zip([float(x) for x in ns_l], [float(x) for x in nb_l]))
                writer.writerows(zip([float(x) for x in e],    [float(x) for x in e_err]))
                writer.writerows(zip([float(x) for x in b],    [float(x) for x in b_err]))
                print(f'created csv at {output_path}/FOM.csv')
        except Exception as err:
            traceback.print_exc()

    return result
