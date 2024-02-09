The notebooks in here will be using a base file and producing plots showing how the parameters affect the output city-by-city.

Hypathia has been generated from detsim for me, so working from Sophronia onwards.

Hypathia in the current dataset sets the SiPM readout window to be 1 us, so extrapolate from that.

### SOPHRONIA

parameters of relevance:

- q_thr: same as q_cut in Beersheba. Minimum number of SiPMs per hit.

- rebin: the rebinning across time for the waveforms. *This one is important! it allows you to increase the SiPM readout window*

possible other relevant parameters:

- s1_params: parameters relevant to the s1 signals (I choose to ignore this) 

- s2_params: parameters relevant to the s2 signals (I choose to ignore this)

- rebin_method: idk

- sipm_charge_type: the way the charge is processed I'm assuming? It's raw currently.

- global_reco_algo & global_reco_params: idk

- same_peak: idk



### BEERSHEBA

To be written

### ISAURA

parameters of relevance:

- voxel size:

- others I havent written