import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt

files = glob('data' + "/*.h5")
x = pd.read_hdf(f'{files[0]}', 'RECO/Events')
print(x)

numb_hits = []
E_hits    = []
Q_check   = []

for f in tqdm(files):
    try:
        x = pd.read_hdf(f'{f}', 'RECO/Events')

        for evt, df in x.groupby('event'):

            numb_hits.append(len(df))
            E_hits.append(df.Ec.sum())
            Q_check.append(df.Q.values)
    except:
        print(f'File {f} broke')

#Q_check = np.concatenate(Q_check)
#plt.hist(Q_check)
#plt.xlabel('hit charge (PEs)')
#plt.show()
#print(Q_check.min())


plt.hist(numb_hits, bins = 50, range=[0, 200])
plt.xlabel('number of hits')
plt.ylabel('counts')
plt.savefig('plots/numb_hits_far.pdf')
plt.show()


plt.hist(numb_hits, bins = 50, range=[0, 50])
plt.xlabel('number of hits')
plt.ylabel('counts')
plt.savefig('plots/numb_hits_close.pdf')
plt.show()

print(f'Minimum number of hits for Kr event: {min(numb_hits)}')
print(f'Energy for that event: {E_hits[np.argmin(np.array(numb_hits))]:.2f} MeV')

plt.scatter(numb_hits, E_hits)
plt.xlabel('Number of hits')
plt.ylabel('Event energy (MeV)')
plt.xlim([0, 200])
plt.ylim([0, 0.1])
plt.savefig('plots/numb_against_E.pdf')
plt.show()
