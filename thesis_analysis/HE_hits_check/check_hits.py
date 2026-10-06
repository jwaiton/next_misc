import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt

files = glob('data' + "/*.h5")
x = pd.read_hdf(f'{files[0]}', 'Tracking/Tracks')
print(x)
numb_hits = []
E_hits    = []

for f in tqdm(files):
    try:
        x = pd.read_hdf(f'{f}', 'Tracking/Tracks')

        for evt, df in x.groupby('event'):
            numb_hits.append(df.numb_of_hits.values)
            E_hits.append(df.energy.values)
    except Exception as e:
        print(f'File {f} broke')
        print(e)

E_hits    = np.concatenate(E_hits)
numb_hits = np.concatenate(numb_hits)
#plt.hist(Q_check)
#plt.xlabel('hit charge (PEs)')
#plt.show()
#print(Q_check.min())

#
plt.hist(numb_hits, bins = 50)
plt.xlabel('number of hits')
plt.ylabel('counts')
plt.savefig('plots/numb_hits_far.pdf')
plt.show()
#
#
#plt.hist(numb_hits, bins = 50, range=[0, 50])
#plt.xlabel('number of hits')
#plt.ylabel('counts')
#plt.savefig('plots/numb_hits_close.pdf')
#plt.show()
#
print(f'Minimum number of hits for Kr event: {min(numb_hits)}')
print(f'Energy for that event: {E_hits[np.argmin(np.array(numb_hits))]:.2f} MeV')
#
plt.scatter(numb_hits, E_hits)
plt.xlabel('Number of hits')
plt.ylabel('Event energy (MeV)')
plt.savefig('plots/numb_against_E.pdf')
plt.show()

plt.scatter(numb_hits, E_hits)
plt.xlabel('Number of hits')
plt.ylabel('Event energy (MeV)')
plt.xlim([0, 25])
plt.ylim([0, 0.25])
plt.show()
