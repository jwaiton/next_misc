#############################################################
#
#          FOM function 
#
# jwaiton 170426
#
#############################################################


import csv
import numpy as np
import traceback
import matplotlib.pyplot as plt

import plotting_funcs as plotf
import fitting_funcs as fitf
import cutting_funcs as cutf

import probfit

def FOM(data, signal_func, background_func, cut_list = None, seeds = None, fitting_info = None, plot = False, output_path = None, label = None):
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
        _seeds     :  dict
                            Seeds for combined signal and background
                            function, should be {'label': value}
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
    if fitting_info is None:
        print('Fitting info not provided, setting defaults')
        fitting_info = dict(
                            fit_range = (1.4, 1.8),
                            bins      = 80)

    if cut_list is None:
        cut_list = np.linspace(0, 0.5, 31)

    # generate functions using probfit
    sig_norm       = probfit.Normalized(signal_func, fitting_info['fit_range']) 
    sig_norm_ext   = probfit.Extended(sig_norm, extname = 'Ns')

    bck_norm       = probfit.Normalized(background_func, fitting_info['fit_range'])
    bck_norm_ext   = probfit.Extended(bck_norm, extname = 'Nb')
    
    # create initial lists
    e, b, ns_l, nb_l, fom, fom_err, e_err, b_err = [[] for _ in range(8)]

    
    # collect smaller dataset for gaussian fitting
    # hardcoded around the peak at ~1.6
    fresh_data = data.copy(deep = True)
    gauss_hdst = cutf.energy_cuts(fresh_data, 1.55, 1.65) 
    if plot:
       plotf.plot_hist(gauss_hdst, binning = 20, output = True, log = False, title = 'Gaussian fit bins') 
    mu, sigma = fitf.gaussian_fit(gauss_hdst, fitting_info)
    print(f'Gaussian fixed at mu {mu} sigma {sigma}')
    # update the seeds to match this
    seeds.update({'loc' : mu, 'scale' : sigma})
    
    # begin the fitting madness
    for i in range(len(cut_list)):
        try:
            # generate new dataset
            blob_data = fresh_data[(fresh_data['eblob2'] > cut_list[i])]
            print('=' * 30, flush = True)
            print(f'Blob cut: {cut_list[i]} MeV', flush = True)
            print('=' * 30, flush = True)

            # apply the signal fit here
            ns, nb = fitf.sb_fit(blob_data, sig_norm_ext, bck_norm_ext, fitting_info, seeds)
            
            # boring appending
            ns_l.append(ns)
            nb_l.append(nb)
            
            print(f"Signal events: {ns}\nBackground events: {nb}\n", flush = True)
            print(f"Total events by addition: {ns+nb}\nTotal events by row counting: {len(blob_data.index)}", flush = True)
            
            # fom calc
            e_check   = ns/ns_l[0]
            b_check   = nb/nb_l[0]
            fom_check = e_check/np.sqrt(b_check)

            e.append(  e_check)
            b.append(  b_check)
            fom.append(fom_check)

            e_err.append(fitf.ratio_error(e_check, 
                                          ns, ns_l[0],
                                          np.sqrt(ns), np.sqrt(ns_l[0])))

            b_err.append(fitf.ratio_error(b_check, 
                                          nb, nb_l[0],
                                          np.sqrt(nb), np.sqrt(nb_l[0])))

            fom_err.append(fitf.fom_error(e_check, b_check, e_err[i], b_err[i]))


            del blob_data, ns, nb
            print(f"FOM: {fom_check} +/- {fom_err}")
            print('=' * 30) 
            

        except Exception as exc:
            print(f'FIT BROKE!')
            print(traceback.format_exc())
            ns_l    .append(-9999)
            nb_l    .append(-9999)
            e       .append(-9999)
            b       .append(-9999)
            fom     .append(-9999)
            e_err   .append(-9999)
            b_err   .append(-9999)
            fom_err .append(-9999)
            try:
                del blob_data, holder_sb
            except:
                print('attempted to delete the current iteration data, but failed') 


    print(f'FOM list:\n{fom}\n', flush = True)

    print(f'CUT list:\n{cut_list}\n')

    # wipe a plot if it exists, if not no worries
    try:
        plt.show()
    except:
        pass

    try:
        # plot and write
        plt.errorbar(cut_list, fom, y_err = fom_err, label = 'FIT', linestyle = 'dashed')
        plt.legend()
        plt.title('FOM {label}')
        plt.xlabel('Blob 2 energy threshold (MeV)')
        plt.ylabel('FOM')
        plt.savefig(f"{'/'.join(output_path.split('/')[:-1])}/FOM_fit.png") 
        plt.close()
    except Exception as err:
        print(err)
    
    with open(f'{output_path}/FOM.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(zip(cut_list, fom))
        writer.writerows(zip(cut_list, fom_err))
        writer.writerows(zip(ns_l, nb_l))
        writer.writerows(zip(e, e_err))
        writer.writerows(zip(b, b_err))


    return dict(fom = fom, cut_list = cut_list, nb_l = nb_l, ns_l = ns_l, e = e, b = b, e_err = e_err, b_err = b_err, fom_err = fom_err)